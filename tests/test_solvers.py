"""Solver tests, each anchored to either an analytic result or the other solver.

Every check here corresponds to a bug that was actually shipped at some point, so they are written
against physics rather than against whatever the code currently returns.
"""

import numpy as np
import pytest
from conftest import analytic_levels, harmonic

from bloch_schrodinger.fdsolver import FDSolver
from bloch_schrodinger.potential import Potential
from bloch_schrodinger.pwsolver import PWSolver

L = 4.0


def cosine_cell(resolution=(32, 32), asymmetric=True):
    """A smooth, deliberately non-centrosymmetric lattice: a mirrored mode would show up."""
    p = Potential(unitvecs=[[L, 0], [0, L]], resolution=resolution, v0=0)
    V = 5 * np.cos(2 * np.pi * p.x / L) + 2 * np.cos(2 * np.pi * p.y / L)
    if asymmetric:
        V = V + 3 * np.cos(4 * np.pi * p.x / L + 0.7)
    p.set(V)
    return p


def skewed_cell(resolution=(32, 32)):
    """A hexagonal cell, where the lattice vectors are not orthogonal."""
    a = 4.0
    p = Potential(
        unitvecs=[[a, 0], [a / 2, a * 3**0.5 / 2]], resolution=resolution, v0=0
    )
    b1 = 2 * np.pi * np.array([1 / a, -1 / (a * 3**0.5)])
    b2 = 2 * np.pi * np.array([0, 2 / (a * 3**0.5)])
    p.set(sum(4 * np.cos(G[0] * p.x + G[1] * p.y + 0.5) for G in (b1, b2, b1 + b2)))
    return p


# --------------------------------------------------------------------------------------
# Analytic spectra
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "omegas, resolution, extent, rtol",
    [
        ((2.0,), (400,), 12.0, 2e-3),
        ((2.0, 3.0), (70, 70), 10.0, 2e-2),
    ],
)
def test_fdsolver_reproduces_harmonic_oscillator(omegas, resolution, extent, rtol):
    """The harmonic trap's levels are sum_i omega_i (n_i + 1/2), with alpha = 1/2."""
    n_dims = len(omegas)
    unitvecs = (np.eye(n_dims) * extent).tolist()
    pot = harmonic(Potential(unitvecs=unitvecs, resolution=resolution, v0=0), omegas)

    n_levels = 3
    eigva, _ = FDSolver(pot, 0.5).solve(n_eigva=n_levels, verbose=False)
    got = np.sort(np.atleast_1d(eigva.squeeze().values))[:n_levels]

    np.testing.assert_allclose(got, analytic_levels(omegas, n_levels), rtol=rtol)


# --------------------------------------------------------------------------------------
# The two solvers must agree, in every dimensionality and on a skewed lattice
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_potential",
    [
        pytest.param(lambda: cosine_cell(), id="orthogonal"),
        pytest.param(lambda: skewed_cell(), id="skewed"),
    ],
)
def test_plane_wave_and_finite_difference_agree(make_potential):
    pot = make_potential()
    e_pw, _ = PWSolver(pot, 0.5, 60).solve(1, verbose=False)
    e_fd, _ = FDSolver(pot, 0.5).solve(1, verbose=False)
    assert abs(float(e_pw) - float(e_fd)) < 2e-2


def test_plane_wave_1d_and_3d_agree_with_finite_difference():
    p1 = Potential(unitvecs=[[L]], resolution=(48,), v0=0)
    p1.set(5 * np.cos(2 * np.pi * p1.coords[0] / L) + 3 * np.cos(4 * np.pi * p1.coords[0] / L + 0.7))
    assert abs(float(PWSolver(p1, 0.5, 60).solve(1, verbose=False)[0])
               - float(FDSolver(p1, 0.5).solve(1, verbose=False)[0])) < 1e-2

    p3 = Potential(unitvecs=[[L, 0, 0], [0, L, 0], [0, 0, L]], resolution=(16, 16, 16), v0=0)
    x, y, z = p3.coords
    p3.set(4 * np.cos(2 * np.pi * x / L) + 2 * np.cos(2 * np.pi * y / L) + 2 * np.cos(2 * np.pi * z / L))
    assert abs(float(PWSolver(p3, 0.5, 25).solve(1, verbose=False)[0])
               - float(FDSolver(p3, 0.5).solve(1, verbose=False)[0])) < 1e-1


def test_compute_u_matches_finite_difference_profile():
    """Regression: compute_u once offset x to the cell origin but not y, displacing every mode."""
    pot = cosine_cell(resolution=(48, 48))
    pw = PWSolver(pot, 0.5, 60)
    _, coeffs = pw.solve(1, verbose=False)
    _, eigve_fd = FDSolver(pot, 0.5).solve(1, verbose=False)

    u_pw = np.abs(pw.compute_u(coeffs.squeeze())) ** 2
    u_pw = (u_pw / u_pw.max()).transpose("a1", "a2").values
    u_fd = np.abs(eigve_fd.squeeze()) ** 2
    u_fd = (u_fd / u_fd.max()).transpose("a1", "a2").values

    assert np.abs(u_pw - u_fd).max() < 0.02


# --------------------------------------------------------------------------------------
# The fast paths must agree with the slow ones they replaced
# --------------------------------------------------------------------------------------


def test_compute_u_fft_matches_explicit_sum():
    """The FFT route is an optimisation, not a different answer."""
    pot = cosine_cell()
    pw = PWSolver(pot, 0.5, 80)
    _, coeffs = pw.solve(2, verbose=False)
    coeffs = coeffs.squeeze()

    fast = pw.compute_u(coeffs)
    slow = pw.compute_u(coeffs, coords=pot.coords, vectorized=True).transpose(*fast.dims)

    rel = float(np.abs(fast - slow).max()) / float(np.abs(slow).max())
    assert rel < 1e-12
    # The cartesian coords have to survive, the plotting functions read them
    for name in ("a1", "a2", "x", "y"):
        assert name in fast.coords


def test_dense_and_sparse_diagonalisation_agree():
    """dense_limit only picks an algorithm; it must not change the spectrum."""
    pot = cosine_cell()
    e_dense, v_dense = PWSolver(pot, 0.5, 80, dense_limit=10**9).solve(3, verbose=False)
    e_sparse, v_sparse = PWSolver(pot, 0.5, 80, dense_limit=0).solve(3, verbose=False)

    np.testing.assert_allclose(
        e_dense.values, e_sparse.transpose(*e_dense.dims).values, atol=1e-10
    )
    a = v_dense.isel(band=0).values
    b = v_sparse.transpose(*v_dense.dims).isel(band=0).values
    overlap = abs(np.vdot(a, b)) / np.linalg.norm(a) / np.linalg.norm(b)
    assert overlap == pytest.approx(1.0, abs=1e-8)


def test_compute_u_is_normalised():
    pot = cosine_cell()
    pw = PWSolver(pot, 0.5, 80)
    _, coeffs = pw.solve(2, verbose=False)
    u = pw.compute_u(coeffs.squeeze())
    norm = (np.abs(u) ** 2).sum(["a1", "a2"]) * pot.get_dS()
    np.testing.assert_allclose(norm.values, 1.0, rtol=1e-10)


# --------------------------------------------------------------------------------------
# Basis truncation
# --------------------------------------------------------------------------------------


def test_large_cutoff_does_not_run_off_the_transform():
    """Regression: G-G' used to index past the end of fV and raise IndexError."""
    pot = cosine_cell()
    pw = PWSolver(pot, 0.5, 400)
    assert pw.connect.min() >= 0
    assert int(pw.connect.max()) < min(pw.n_a)


def test_aliasing_warning_only_for_unresolved_potentials():
    smooth = cosine_cell(asymmetric=False)
    with warnings_as_errors():
        PWSolver(smooth, 0.5, 2000)  # band-limited: the wrap is harmless, stay quiet

    rough = Potential(unitvecs=[[L, 0], [0, L]], resolution=(32, 32), v0=0)
    rough.set(-20 * np.exp(-((rough.x) ** 2 + (rough.y) ** 2) / 0.02))
    with pytest.warns(UserWarning, match="E_lim"):
        PWSolver(rough, 0.5, 400)


def test_asking_for_more_bands_than_basis_vectors_is_an_error():
    pot = cosine_cell()
    pw = PWSolver(pot, 0.5, 20)
    with pytest.raises(ValueError, match="plane wave basis"):
        pw.solve(pw.nGs + 1, verbose=False)


def warnings_as_errors():
    import warnings
    from contextlib import contextmanager

    @contextmanager
    def ctx():
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            yield

    return ctx()

