"""Progress-reporting tests.

Two things here are easy to get wrong and invisible when they are, so each has a test anchored to
it rather than to current output.

The first is ordering. 'parallel_map' collects results through joblib's
'return_as="generator_unordered"', which hands tasks back in whatever order they finish, and then
puts them back by index. If that re-indexing were dropped the results would silently permute --
every array in a parameter sweep landing at the wrong point, with no error anywhere.

The second is silence. Bars are written to stderr, so a bar that ignores 'verbose=False' cannot be
captured away by the caller; it corrupts any other output sharing the stream. This suite itself
used to pass 'verbose=False' to FDSolver.solve, which had no such parameter and swallowed it in
'**kwargs', so the solvers are checked here for actually honouring it.
"""

import io
import time
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pytest

from bloch_schrodinger.fdsolver import FDSolver
from bloch_schrodinger.potential import Potential, create_parameter
from bloch_schrodinger.progress import bar, parallel_map
from bloch_schrodinger.pwsolver import PWSolver

L = 4.0


def _times_ten(i, factor):
    """Deliberately slowest-first, so completion order is the reverse of input order."""
    time.sleep((10 - i) * 0.01)
    return i * factor


ARGS = [(i, 10) for i in range(10)]
EXPECTED = [i * 10 for i in range(10)]


def sweep_potential(n_points=4, resolution=(24, 24)):
    """A small lattice with one parameter dimension, so a solve has several matrices to do."""
    depth = create_parameter("depth", np.linspace(4.0, 6.0, n_points))
    p = Potential(unitvecs=[[L, 0], [0, L]], resolution=resolution, v0=0)
    p.set(depth * np.cos(2 * np.pi * p.x / L) + 2 * np.cos(2 * np.pi * p.y / L))
    return p


# --------------------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_parallel_map_returns_results_in_input_order(n_jobs):
    """The tasks finish in reverse, so an unindexed collect would come back reversed."""
    err = io.StringIO()
    with redirect_stderr(err):
        got = parallel_map(
            _times_ten, ARGS, n_jobs=n_jobs, desc="Testing", unit="task"
        )
    assert got == EXPECTED


def test_parallel_map_handles_an_empty_task_list():
    with redirect_stderr(io.StringIO()):
        assert parallel_map(_times_ten, [], n_jobs=2, desc="Testing", unit="task") == []


# --------------------------------------------------------------------------------------
# Labelling, and not fighting joblib for the stream
# --------------------------------------------------------------------------------------


def test_parallel_map_bar_is_labelled_and_joblib_stays_quiet():
    """joblib's own verbose output and a tqdm bar cannot share stderr, so it must be off."""
    err = io.StringIO()
    with redirect_stderr(err):
        parallel_map(_times_ten, ARGS, n_jobs=4, desc="Testing", unit="task")
    out = err.getvalue()
    assert "Testing" in out
    assert "task/s" in out or "s/task" in out
    assert "[Parallel(" not in out


def test_bar_requires_a_label():
    """A bar with no desc/unit is the thing the wrapper exists to prevent."""
    with pytest.raises(TypeError, match="unit"):
        bar(range(3), desc="only desc")
    with pytest.raises(TypeError, match="desc"):
        bar(range(3), unit="only unit")


# --------------------------------------------------------------------------------------
# verbose=False is silent
# --------------------------------------------------------------------------------------


def test_bar_and_parallel_map_write_nothing_when_not_verbose():
    err, out = io.StringIO(), io.StringIO()
    with redirect_stderr(err), redirect_stdout(out):
        for _ in bar(range(5), desc="Quiet", unit="x", verbose=False):
            pass
        got = parallel_map(
            _times_ten, ARGS, n_jobs=2, desc="Quiet", unit="task", verbose=False
        )
    assert err.getvalue() == ""
    assert out.getvalue() == ""
    assert got == EXPECTED


@pytest.mark.parametrize("parallel", [False, True])
def test_fdsolver_honours_verbose_false(parallel):
    """FDSolver.solve had no verbose parameter, so **kwargs swallowed it and it printed anyway."""
    pot = sweep_potential()
    err, out = io.StringIO(), io.StringIO()
    with redirect_stderr(err), redirect_stdout(out):
        FDSolver(pot, 0.5).solve(1, parallel=parallel, n_cores=2, verbose=False)
    assert err.getvalue() == ""
    assert out.getvalue() == ""


@pytest.mark.parametrize("parallel", [False, True])
def test_pwsolver_honours_verbose_false(parallel):
    pot = sweep_potential()
    err, out = io.StringIO(), io.StringIO()
    with redirect_stderr(err), redirect_stdout(out):
        pw = PWSolver(pot, 0.5, 40)
        eigva, eigve = pw.solve(1, parallel=parallel, n_cores=2, verbose=False)
        pw.compute_u(eigve, coords=list(pot.coords), verbose=False)
    assert err.getvalue() == ""
    assert out.getvalue() == ""


# --------------------------------------------------------------------------------------
# Parallelizing must not change the answer
# --------------------------------------------------------------------------------------


def test_fdsolver_parallel_matches_serial():
    """Guards the re-indexing end to end: a permutation here would misplace whole eigenvectors."""
    pot = sweep_potential()
    serial_va, serial_ve = FDSolver(pot, 0.5).solve(2, n_cores=2, verbose=False)
    par_va, par_ve = FDSolver(pot, 0.5).solve(
        2, parallel=True, n_cores=2, verbose=False
    )
    np.testing.assert_allclose(serial_va.values, par_va.values)
    np.testing.assert_allclose(np.abs(serial_ve.values), np.abs(par_ve.values))
