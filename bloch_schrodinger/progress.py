"""Progress reporting shared by every solver.

The two entry points here are 'bar', for a loop this process runs itself, and 'parallel_map',
for a loop spread over joblib workers. Both take the same 'desc'/'unit'/'verbose' triple, so a
solver reads the same whether or not it is parallelized, and a single 'verbose=False' silences
the lot.

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

    joblib's own 'verbose' is pinned to 0: its "Done 12 tasks | elapsed 3.1s" lines and a tqdm
    bar fight over the same stream, and only one of them can win. The results come back through
    'return_as="generator_unordered"' and are put back in order here by index, so the bar counts
    tasks as they actually finish - with an ordered return a single slow early task freezes the
    bar while the rest of the sweep completes behind it.

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
    n = len(args_list)

    def indexed(i, args):
        return i, func(*args)

    pool = Parallel(n_jobs=n_jobs, return_as=_RETURN_AS, verbose=0)
    results = [None] * n
    with bar(total=n, desc=desc, unit=unit, verbose=verbose) as pbar:
        for i, result in pool(delayed(indexed)(i, a) for i, a in enumerate(args_list)):
            results[i] = result
            pbar.update(1)
    return results
