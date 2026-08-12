"""Wannier and plotting tests.

Like the solver suite, these are anchored to conditions that must hold rather than to current output:
the Marzari-Vanderbilt stencil identity, normalisation, and the dimensionality contracts.
"""

import numpy as np
import pytest
import xarray as xr

from bloch_schrodinger.plotting import (
    _isosurface_mesh,
    _spatial_dims,
    _to_orthogonal,
    plot_isosurface,
)
from bloch_schrodinger.potential import Potential
from bloch_schrodinger.wannier import Wannier

L = 4.0


def cosine_1d(resolution=(48,)):
    p = Potential(unitvecs=[[L]], resolution=resolution, v0=0)
    p.set(-20 * np.cos(2 * np.pi * p.coords[0] / L))
    return p


def honeycomb(resolution=(32, 32)):
    kl, a = 2, 4 * np.pi / 3**1.5 / 2
    a1 = np.array([3 * a / 2, -(3**0.5) * a / 2])
    a2 = np.array([3 * a / 2, 3**0.5 * a / 2])
    ks = [
        kl * np.array([-(3**0.5) / 2, 1 / 2]),
        kl * np.array([3**0.5 / 2, 1 / 2]),
        kl * np.array([0, -1]),
    ]
    p = Potential(unitvecs=[a1, a2], resolution=resolution, v0=0)
    dirs = [k[0] * (p.x - a1[0]) + k[1] * p.y for k in ks]
    for i in range(3):
        p.add(2 * (-6) * np.cos((dirs[i - 1] - dirs[i]) - 2 * np.pi / 3) / 2)
        p.add(2 * (-6) * np.cos((dirs[i - 1] - dirs[i]) + 2 * np.pi / 3) / 2)
    a1s = np.array([-1, 3**0.5]) * 2 * np.pi / 3 / a
    a2s = np.array([1, 3**0.5]) * 2 * np.pi / 3 / a
    return p, [a1s, a2s], a


# --------------------------------------------------------------------------------------
# The stencil identity the whole spread functional rests on
# --------------------------------------------------------------------------------------


def test_stencil_condition_1d():
    w = Wannier(cosine_1d(), 0.5, [np.array([2 * np.pi / L])], (15,))
    got = float((w.weights * w.neighbors.sel(kxy=0) ** 2).sum("b"))
    assert got == pytest.approx(1.0, abs=1e-10)


def test_stencil_condition_2d():
    pot, rec, _ = honeycomb()
    w = Wannier(pot, 0.5, rec, (9, 9))
    for c1 in range(2):
        for c2 in range(2):
            got = float(
                (w.weights * w.neighbors.sel(kxy=c1) * w.neighbors.sel(kxy=c2)).sum("b")
            )
            assert got == pytest.approx(1.0 if c1 == c2 else 0.0, abs=1e-10)


# --------------------------------------------------------------------------------------
# Dimensionality contracts
# --------------------------------------------------------------------------------------


def test_wannier_rejects_3d_with_a_clear_message():
    p = Potential(unitvecs=[[4, 0, 0], [0, 4, 0], [0, 0, 4]], resolution=(8, 8, 8), v0=0)
    p.set(p.coords[0] * 0)
    with pytest.raises(ValueError, match="1D and 2D"):
        Wannier(p, 0.5, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], (5, 5, 5))


def test_wannier_rejects_mismatched_reciprocal_vectors():
    with pytest.raises(ValueError, match="reciprocal vector"):
        Wannier(cosine_1d(), 0.5, [[1, 0], [0, 1]], (15, 15))


def test_plot_isosurface_rejects_non_3d():
    p = Potential(unitvecs=[[4, 0], [0, 4]], resolution=(16, 16), v0=0)
    p.set(p.x**2 + p.y**2)
    with pytest.raises(ValueError, match="3 spatial axes"):
        plot_isosurface(p.V)


# --------------------------------------------------------------------------------------
# 1D Wannier functions, end to end
# --------------------------------------------------------------------------------------


def test_wannier_1d_is_normalised_and_localised():
    pot = cosine_1d()
    w = Wannier(pot, 0.5, [np.array([2 * np.pi / L])], (9,), method="pw")
    U = w.solve(n_wannier=1, centers=[[0.0]], blockwargs={"E_lim": 150})
    tiled, wf = w.compute_wannier(U_mnk=U, bounds=[(-2, 3)])

    norm = float((np.abs(wf) ** 2).isel(n=0).sum("a1")) * tiled.get_dS()
    assert norm == pytest.approx(1.0, rel=1e-8)

    profile = (np.abs(wf) ** 2).isel(n=0).transpose("a1").values
    xs = np.asarray(wf.x.values)
    # It must sit in the well at the origin, and die away from it
    assert abs(xs[profile.argmax()]) < 0.1
    assert profile.max() / profile[0] > 1e6


# --------------------------------------------------------------------------------------
# The spread minimization
# --------------------------------------------------------------------------------------


def _prepared_wannier():
    pot, rec, a = honeycomb()
    w = Wannier(pot, 0.5, rec, (7, 7), method="pw")
    w.n_wannier = [0, 2]
    w.compute_bloch(E_lim=300)
    w.nbands = 2
    return w, [[-a / 2, 0], [a / 2, 0]]


def test_seed_makes_the_minimization_reproducible():
    """The initial guess is randomized, so without a seed two identical runs disagree."""
    pot, rec, a = honeycomb()
    w = Wannier(pot, 0.5, rec, (7, 7), method="pw")
    centers = [[-a / 2, 0], [a / 2, 0]]
    kwargs = dict(n_wannier=2, centers=centers, blockwargs={"E_lim": 300})

    first = w.solve(seed=1234, **kwargs)
    second = w.solve(seed=1234, **kwargs)
    assert float(np.abs(first - second).max()) == 0.0


def test_conjugate_gradient_reaches_the_same_minimum_as_steepest_descent():
    w, centers = _prepared_wannier()
    _, sd = w.compute_U_mnk(
        {}, centers, 1e-7, method="sd", rng=np.random.default_rng(7), return_info=True
    )
    _, cg = w.compute_U_mnk(
        {}, centers, 1e-7, method="cg", rng=np.random.default_rng(7), return_info=True
    )
    assert cg["spread"] == pytest.approx(sd["spread"], rel=1e-4)
    assert cg["iterations"] < sd["iterations"]


def test_minimization_respects_max_iter_and_says_so():
    """Regression: the loop had no cap at all and could run indefinitely."""
    w, centers = _prepared_wannier()
    with pytest.warns(UserWarning, match="without meeting tol"):
        _, info = w.compute_U_mnk(
            {},
            centers,
            1e-15,
            max_iter=4,
            rng=np.random.default_rng(7),
            return_info=True,
        )
    assert info["iterations"] == 4
    assert not info["converged"]


def test_unknown_minimization_method_is_rejected():
    w, centers = _prepared_wannier()
    with pytest.raises(ValueError, match="'cg' or 'sd'"):
        w.compute_U_mnk({}, centers, 1e-7, method="newton")


def test_initial_step_matches_the_marzari_vanderbilt_value():
    w, _ = _prepared_wannier()
    assert w.initial_step() == pytest.approx(1 / (4 * float(w.weights.sum())))


# --------------------------------------------------------------------------------------
# Cyclic colour fields
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cyclic, tolerance",
    [
        pytest.param(True, 0.5, id="cyclic-wraps-smoothly"),
        pytest.param(False, None, id="linear-tears-at-the-seam"),
    ],
)
def test_isosurface_cyclic_colour_does_not_tear_at_the_seam(cyclic, tolerance):
    """Regression: a phase interpolated linearly runs the long way round at the +/-pi seam.

    The field has an exactly known phase, a ramp in x, so every sampled value can be checked against
    the truth. The non-cyclic case is asserted to be bad, so the test cannot pass vacuously.
    """
    p = Potential(unitvecs=[[10, 0, 0], [0, 10, 0], [0, 0, 10]], resolution=(32, 32, 32), v0=0)
    p.set(sum(c**2 for c in p.coords))
    x, y, z = p.coords
    k = 2.5
    psi = np.exp(-(x**2 + y**2 + z**2) / 8) * np.exp(1j * k * x)
    phase = xr.ufuncs.angle(psi)

    spatial = _spatial_dims(np.abs(psi))
    volume = _to_orthogonal(np.abs(psi), spatial)
    period = 2 * np.pi
    # plot_isosurface carries a cyclic field around the unit circle before regridding
    colour = _to_orthogonal(
        np.exp(2j * np.pi * phase / period) if cyclic else phase, spatial
    )

    Xc, Yc, Zc = volume.x.values, volume.y.values, volume.z.values
    spacing = (Xc[1] - Xc[0], Yc[1] - Yc[0], Zc[1] - Zc[0])
    origin = (Xc[0], Yc[0], Zc[0])
    level = float(volume.min()) + 0.35 * float(volume.max() - volume.min())

    verts, _, intensity = _isosurface_mesh(
        volume, colour, level, spacing, origin, period if cyclic else None
    )

    want = np.angle(np.exp(1j * k * verts[:, 0]))
    err = np.abs(np.angle(np.exp(1j * (intensity - want))))

    if cyclic:
        assert err.max() < tolerance
    else:
        # Without the unit-circle detour the seam inverts the phase outright
        assert err.max() > 2.0
