"""Progress reporting shared by every solver.

The entry points here are 'bar', for a loop this process runs itself, and 'parallel_map' /
'parallel_imap', for a loop spread over joblib workers. All three take the same
'desc'/'unit'/'verbose' triple, so a solver reads the same whether or not it is parallelized, and
a single 'verbose=False' silences the lot.

'parallel_map' collects the sweep and returns it; 'parallel_imap' streams '(index, result)' as
each task lands. Prefer the streaming one whenever a single result is a full grid or a stack of
them -- collecting those means holding the entire sweep in memory at once.

tqdm is taken from 'tqdm.auto' rather than plain 'tqdm': the package already depends on
ipywidgets, so a notebook gets a widget bar - which no other writer to stderr can corrupt - and
a terminal gets the usual ANSI one.
"""

from joblib import Parallel, delayed
from tqdm.auto import tqdm

# 'generator_unordered' is what lets a bar count genuine completions; it landed in joblib 1.4.
# Resolving it once here means an older joblib degrades to a bar that fills at the end instead
# of raising. Both settings hand back the same (index, result) pairs, so there is one consumption
# path in parallel_map either way.
try:
    Parallel(return_as="generator_unordered")
    _RETURN_AS = "generator_unordered"
except ValueError:  # joblib < 1.4
    _RETURN_AS = "list"


def bar(
    iterable=None,
    *,
    total: float | None = None,
    desc: str,
    unit: str,
    verbose: bool = True,
    leave: bool = True,
    position: int | None = None,
) -> tqdm:
    """A tqdm bar carrying a label.

    Args:
        iterable: What to iterate over, or None to drive the bar by hand with 'update'.
        total (float, optional): Size of the job, only needed when 'iterable' has no length or
        when the bar counts something other than iterations.
        desc (str): What the bar is measuring, shown to its left.
        unit (str): What one unit of progress is, which is also what the rate is quoted in.
        verbose (bool, optional): False disables the bar in place, so the call site keeps a
        single code path. Defaults to True.
        leave (bool, optional): Whether the finished bar stays on screen. Pass False for a bar
        nested inside another one, so the screen does not fill with dead bars. Defaults to True.
        position (int, optional): Line to draw on, for nested bars. Ignored by the notebook
        widget. Defaults to None.

    Returns:
        tqdm: The bar, usable as an iterator or as a context manager.
    """
    # 'desc' and 'unit' are keyword-only and have no defaults on purpose: a bar without a label
    # is the thing this module exists to prevent, so forgetting one is a TypeError.
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        disable=not verbose,
        leave=leave,
        position=position,
    )


def parallel_imap(
    func,
    args_list,
    *,
    n_jobs: int,
    desc: str,
    unit: str,
    verbose: bool = True,
    longest_first: bool = False,
):
    """Yield '(index, result)' on joblib as each task finishes, keeping none of them.

    This is the streaming half of 'parallel_map'. The difference matters whenever a result is
    large: 'parallel_map' has to hold the whole sweep in memory before it can return any of it,
    where a solver whose result is a stack of full grids only ever needs one at a time. Handing
    each result over as it lands lets the caller store it and drop it, which is the difference
    between one run's worth of memory and the sweep's.

    joblib's own 'verbose' is pinned to 0: its "Done 12 tasks | elapsed 3.1s" lines and a tqdm
    bar fight over the same stream, and only one of them can win. Results come back through
    'return_as="generator_unordered"', so the bar counts tasks as they actually finish - with an
    ordered return a single slow early task freezes the bar while the rest completes behind it.
    The index is yielded alongside the result because that ordering is lost.

    Args:
        func (Callable): The work to do, called as 'func(*args)' for each entry.
        args_list (Sequence[tuple]): One tuple of positional arguments per task.
        n_jobs (int): Cores to spread the tasks over, -1 for all of them.
        desc (str): What the bar is measuring.
        unit (str): What one task is, e.g. "run" or "matrix".
        verbose (bool, optional): Whether to show the bar. Defaults to True.
        longest_first (bool, optional): Dispatch the list back to front. When tasks have very
        unequal costs and the expensive ones sit at the end of the sweep -- which is the usual
        shape, since a parameter is usually swept from its mild end to its severe one -- leaving
        them last means the longest task starts last and every other worker waits on it. Handing
        them out first packs the tail instead. Defaults to False, which preserves the order given.

    Yields:
        tuple[int, object]: The index into 'args_list', and that task's result.
    """
    args_list = list(args_list)
    order = range(len(args_list) - 1, -1, -1) if longest_first else range(len(args_list))

    def indexed(i, args):
        return i, func(*args)

    pool = Parallel(n_jobs=n_jobs, return_as=_RETURN_AS, verbose=0)
    with bar(total=len(args_list), desc=desc, unit=unit, verbose=verbose) as pbar:
        for i, result in pool(delayed(indexed)(i, args_list[i]) for i in order):
            # Yielded before the bar moves, so the bar counts results the caller has taken rather
            # than results that are merely computed and still queued.
            yield i, result
            pbar.update(1)


def parallel_map(
    func,
    args_list,
    *,
    n_jobs: int,
    desc: str,
    unit: str,
    verbose: bool = True,
) -> list:
    """Map 'func' over 'args_list' on joblib, showing one bar that advances per completed task.

    A thin collector over 'parallel_imap', kept for the callers whose results are small enough
    that holding the whole sweep costs nothing. Anything returning full grids should use
    'parallel_imap' directly and store as it goes.

    Args:
        func (Callable): The work to do, called as 'func(*args)' for each entry.
        args_list (Sequence[tuple]): One tuple of positional arguments per task.
        n_jobs (int): Cores to spread the tasks over, -1 for all of them.
        desc (str): What the bar is measuring.
        unit (str): What one task is, e.g. "run" or "matrix".
        verbose (bool, optional): Whether to show the bar. Defaults to True.

    Returns:
        list: The results, in the order of 'args_list'.
    """
    args_list = list(args_list)
    results = [None] * len(args_list)
    for i, result in parallel_imap(
        func, args_list, n_jobs=n_jobs, desc=desc, unit=unit, verbose=verbose
    ):
        results[i] = result
    return results
