"""How 'create_map' chooses what to draw, and what it refuses to compute along the way.

The expensive mistakes this module guards against are not wrong pictures but wrong *amounts of
work*: interpolating a whole sweep to relabel its axes, or materialising a moving frame that is
stored as two small factors. Both are invisible in a rendered image and obvious in a task count,
so several of these tests assert on how much was evaluated rather than on what came out.
"""

import numpy as np
import pytest
import xarray as xr

import bloch_schrodinger.plotting as plotting
from bloch_schrodinger.plotting import (
    _aligned_axes,
    _cart_at,
    _cart_factors,
    _rank1_axis,
    _spatial_dims,
    create_map,
)
from bloch_schrodinger.potential import Potential

LATTICE = ["a1", "a2", "a3"]
CART = ["x", "y", "z"]


@pytest.fixture
def aligned3d():
    """An axis-aligned 3-D field, asymmetric so that a transposition cannot pass unnoticed."""
    pot = Potential(unitvecs=[[8, 0, 0], [0, 10, 0], [0, 0, 4]], resolution=(12, 14, 6), v0=0)
    x, y, z = xr.broadcast(pot.x, pot.y, pot.z)
    return (np.exp(-((x - 1) ** 2 / 6 + (y + 2) ** 2 / 9 + z**2 / 3))
            * (1 + 0.4 * np.sin(2 * x))).rename("f")


@pytest.fixture
def moving():
    """A field on a moving grid, stored factored as an invariant lattice times a per-frame scale."""
    pot = Potential(unitvecs=[[8, 0], [0, 10]], resolution=(12, 14), v0=0)
    rho_x, rho_y = xr.broadcast(pot.x, pot.y)
    tv = np.linspace(0, 1, 4)
    t = xr.DataArray(tv, dims="t", coords={"t": tv}, name="t")
    lam = 1 + 0.7 * t
    field = np.exp(-(rho_x**2 + rho_y**2) / 4) * (1 + 0 * t)
    field = field.transpose("t", "a1", "a2").rename("f")
    # dropped exactly as a run saved with store_frame=False does: the frame lives in the factors
    return field.drop_vars(["x", "y"]).assign_coords(
        rho_x=rho_x.drop_vars(["x", "y"], errors="ignore"),
        rho_y=rho_y.drop_vars(["x", "y"], errors="ignore"),
        lambda_x=lam,
        lambda_y=lam * 1.1,
    )


def draw(data, cart_axes, **kw):
    """Build a map, move it once, and hand back the artist's values and mesh."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        sliders, update, _ = create_map(fig, ax, cart_axes, data, "pcolormesh", **kw)
        values = {k: w.value for k, w in sliders.items()}
        update(**values)
        artist = ax.collections[-1]
        return (np.asarray(artist.get_array()).copy(),
                artist._coordinates.copy(), sliders, values)
    finally:
        plt.close(fig)


# --- which path gets taken -------------------------------------------------------------------

def test_rank1_axis_finds_the_only_axis_that_varies():
    v = np.broadcast_to(np.arange(5.0)[:, None, None], (5, 3, 4)).copy()
    assert _rank1_axis(v) == 0
    assert _rank1_axis(np.moveaxis(v, 0, 2)) == 2
    assert _rank1_axis(np.arange(60.0).reshape(5, 3, 4)) is None


def test_axis_aligned_lattice_needs_no_interpolation(aligned3d):
    assert _aligned_axes(aligned3d, _spatial_dims(aligned3d)) == [0, 1, 2]


def test_skewed_lattice_is_still_interpolated():
    a = 2 * 1.064 / 3
    pot = Potential(
        unitvecs=[[a * 3**0.5 / 2, -a / 2], [a * 3**0.5 / 2, a / 2]],
        resolution=(16, 16), v0=0,
    )
    assert _aligned_axes(pot.V, _spatial_dims(pot.V)) is None


def test_permuted_box_reports_the_permutation():
    pot = Potential(unitvecs=[[0, 20], [15, 0]], resolution=(8, 10), v0=0)
    assert _aligned_axes(pot.V, _spatial_dims(pot.V)) == [1, 0]


def test_aligned_field_does_not_call_interp(aligned3d, monkeypatch):
    """The whole point: an already-cartesian lattice must never reach '_to_orthogonal'."""
    called = []
    monkeypatch.setattr(plotting, "_to_orthogonal",
                        lambda *a, **k: called.append(1) or a[0])
    draw(aligned3d, [0, 2])
    assert called == []


# --- what actually gets drawn ----------------------------------------------------------------

@pytest.mark.parametrize("cart_axes", [[0, 1], [0, 2], [1, 2], [1, 0]])
def test_drawn_values_are_the_field_itself(aligned3d, cart_axes):
    """Relabelling is exact, so the artist carries the raw slice, transposition included.

    The values are located by the *cartesian* coordinate, because that is what the axes are indexed
    by once an aligned lattice has been relabelled -- the same thing interpolating used to achieve,
    and what the sliders have to keep saying for a reader to know where they are.
    """
    shown, _, sliders, values = draw(aligned3d, cart_axes)
    plotted = [CART[cart_axes[0]], CART[cart_axes[1]]]
    lattice_of = dict(zip(CART, LATTICE))
    leftover = {
        lattice_of[c]: values[c] for c in CART if c not in plotted and c in values
    }
    truth = aligned3d.sel(leftover, method="nearest")
    truth = truth.transpose(*[lattice_of[c] for c in plotted][::-1])
    assert np.allclose(np.asarray(shown).ravel(), truth.values.ravel())


@pytest.mark.parametrize("cart_axes", [[0, 1], [0, 2], [1, 2]])
def test_leftover_slider_is_named_and_scaled_cartesian(aligned3d, cart_axes):
    """The regression guard: a leftover axis must stay a cartesian coordinate, not a lattice index.

    Drawing an aligned lattice natively is only worth doing if it keeps saying where it is. A
    slider labelled "a3", running over lattice indices, tells a reader nothing about the plane they
    are looking at, and two subplots that name the same axis differently stop sharing one slider.
    """
    _, _, sliders, _ = draw(aligned3d, cart_axes)
    leftover = [CART[i] for i in range(3) if i not in cart_axes]
    for name in leftover:
        assert name in sliders, f"expected a cartesian slider {name!r}, got {sorted(sliders)}"
        coord = aligned3d.coords[name]
        assert np.isclose(sliders[name].min, float(coord.min()))
        assert np.isclose(sliders[name].max, float(coord.max()))
    assert not (set(sliders) & set(LATTICE)), f"lattice sliders leaked: {sorted(sliders)}"


@pytest.mark.parametrize("cart_axes", [[0, 1], [0, 2], [1, 2]])
def test_mesh_spans_the_right_cartesian_coordinate(aligned3d, cart_axes):
    _, mesh, _, _ = draw(aligned3d, cart_axes)
    for k, axis in enumerate(cart_axes):
        coord = aligned3d.coords[CART[axis]]
        # the mesh carries cell edges, so it overshoots the centres by half a cell either side
        span = float(coord.max()) - float(coord.min())
        assert mesh[..., k].min() < float(coord.min()) + 1e-9
        assert mesh[..., k].max() > float(coord.max()) - 1e-9
        assert (mesh[..., k].max() - mesh[..., k].min()) < span * 1.3 + 1e-9


# --- a frame stored as factors ---------------------------------------------------------------

def test_factors_are_read_from_the_array(moving):
    tagged = moving.copy()
    tagged.attrs["cart_factors"] = '{"x": ["rho_x", "lambda_x"]}'
    assert _cart_factors(tagged) == {"x": ["rho_x", "lambda_x"]}
    assert _cart_factors(moving) == {}
    assert _cart_factors(moving, {"x": ("a", "b")}) == {"x": ("a", "b")}


def test_factored_coordinate_matches_the_materialised_one(moving):
    factors = {"x": ("rho_x", "lambda_x"), "y": ("rho_y", "lambda_y")}
    sel = {"t": float(moving.t[2])}
    rebuilt = _cart_at(moving, "x", factors, sel)
    direct = (moving.coords["rho_x"] * moving.coords["lambda_x"]).sel(sel, method="nearest")
    assert np.array_equal(
        rebuilt.transpose("a1", "a2").values, direct.transpose("a1", "a2").values
    )


def test_factored_frame_draws_the_same_as_a_materialised_one(moving):
    factors = {"x": ("rho_x", "lambda_x"), "y": ("rho_y", "lambda_y")}
    materialised = moving.assign_coords(
        x=(moving.coords["rho_x"] * moving.coords["lambda_x"]).transpose("t", "a1", "a2"),
        y=(moving.coords["rho_y"] * moving.coords["lambda_y"]).transpose("t", "a1", "a2"),
    )
    lean_vals, lean_mesh, _, _ = draw(moving, [0, 1], frame=factors)
    full_vals, full_mesh, _, _ = draw(materialised, [0, 1])
    assert np.array_equal(np.asarray(lean_vals), np.asarray(full_vals))
    assert np.array_equal(lean_mesh, full_mesh)


def test_factored_frame_is_recognised_as_moving(moving):
    factors = {"x": ("rho_x", "lambda_x"), "y": ("rho_y", "lambda_y")}
    assert plotting._moving_dims(moving, "x", ["a1", "a2"], factors) == ["t"]
    assert plotting._cart_dims(moving, "x", factors) == ("a1", "a2", "t")


# --- how much gets evaluated -----------------------------------------------------------------

def test_one_frame_costs_a_bounded_number_of_chunks(aligned3d):
    """A whole-array interp or a whole-array transform blows this budget immediately."""
    pytest.importorskip("dask")
    from dask.callbacks import Callback

    # chunked one frame per block, and made big enough that evaluating everything is unmistakable
    stacked = xr.concat(
        [aligned3d * (1 + k) for k in range(40)],
        dim=xr.DataArray(np.arange(40.0), dims="s", coords={"s": np.arange(40.0)}),
    )
    lazy = (abs(stacked) ** 2).chunk({"a1": -1, "a2": -1, "a3": 1, "s": 1})
    n_chunks = int(np.prod([len(c) for c in lazy.data.chunks]))

    class Count(Callback):
        def __init__(self):
            self.n = 0

        def _posttask(self, *a, **k):
            self.n += 1

    with Count() as counter:
        draw(lazy, [0, 1])
    # A frame of this array is 6 chunks, and the coords cost a little on top. The budget is set
    # just above what one frame actually needs rather than merely below the whole array: at
    # n_chunks it would take a 200-fold regression to notice, which is no guard at all.
    assert counter.n <= 30, (
        f"{counter.n} tasks for one frame of a {n_chunks}-chunk array: far more than one frame"
    )


# --- paths that must not regress --------------------------------------------------------------

def test_contour_still_works(aligned3d):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        sliders, update, _ = create_map(fig, ax, [0, 1], aligned3d, "contour")
        update(**{k: w.value for k, w in sliders.items()})
        assert ax.get_children()
    finally:
        plt.close(fig)


def test_moving_grid_mesh_follows_the_frame(moving):
    factors = {"x": ("rho_x", "lambda_x"), "y": ("rho_y", "lambda_y")}
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        sliders, update, _ = create_map(fig, ax, [0, 1], moving, "pcolormesh", frame=factors)
        values = {k: w.value for k, w in sliders.items()}
        values["t"] = float(moving.t[0])
        update(**values)
        first = ax.collections[-1]._coordinates.copy()
        values["t"] = float(moving.t[-1])
        update(**values)
        last = ax.collections[-1]._coordinates.copy()
    finally:
        plt.close(fig)
    # lambda grows by a factor 1.7 over the run, and the mesh must grow with it
    assert last[..., 0].max() > first[..., 0].max() * 1.5
