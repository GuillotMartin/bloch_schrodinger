"""What the drawing helpers put on the canvas, for the paths only notebooks exercised.

'test_create_map.py' pins how 'create_map' *chooses* a path and what it refuses to compute.
This file pins what comes out of the other end, for 'create_line', 'create_quiver' and
'plot_eigenvector', none of which had any coverage at all.

Several tests here assert on bugs rather than on correct behaviour. They are marked as such and
exist so that a refactor cannot quietly change them: a bug that moves is as much a regression as
a feature that breaks, because the notebooks that work around it stop working.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import bloch_schrodinger.plotting as plotting
from bloch_schrodinger.plotting import (
    cmesh_tmpl,
    contour_tmpl,
    create_line,
    create_map,
    create_quiver,
    dashboard,
    plot_eigenvector,
    quiver_tmpl,
)
from bloch_schrodinger.potential import Potential, create_parameter

# Deliberately not square: a square grid lets a transposition through unnoticed, and a
# transposition is exactly what broke 'create_quiver'.
NA1, NA2 = 12, 14

FACTORS = {"x": ("rho_x", "lambda_x"), "y": ("rho_y", "lambda_y")}


# --- fixtures ---------------------------------------------------------------------------------

@pytest.fixture
def aligned2d():
    """An axis-aligned 2-D potential, asymmetric in both value and shape."""
    pot = Potential(unitvecs=[[8, 0], [0, 10]], resolution=(NA1, NA2), v0=0)
    pot.set(pot.x**2 + 2 * pot.y**2 + 0.3 * pot.x)
    return pot


@pytest.fixture
def skewed2d():
    """A triangular lattice, where the cartesian and lattice axes genuinely mix."""
    a = 2 * 1.064 / 3
    pot = Potential(
        unitvecs=[[a * 3**0.5 / 2, -a / 2], [a * 3**0.5 / 2, a / 2]],
        resolution=(NA1, NA2),
        v0=0,
    )
    pot.set(pot.x**2 + 2 * pot.y**2)
    return pot


@pytest.fixture
def aligned3d():
    pot = Potential(unitvecs=[[8, 0, 0], [0, 10, 0], [0, 0, 4]], resolution=(NA1, NA2, 6), v0=0)
    x, y, z = xr.broadcast(pot.x, pot.y, pot.z)
    return np.exp(-((x - 1) ** 2 / 6 + (y + 2) ** 2 / 9 + z**2 / 3)).rename("f")


@pytest.fixture
def param2d():
    """A 2-D field carrying one parameter dimension, the shape most notebooks plot."""
    pot = Potential(unitvecs=[[8, 0], [0, 10]], resolution=(NA1, NA2), v0=0)
    omega = create_parameter("omega", np.linspace(1, 2, 3))
    pot.set((pot.x**2 + 2 * pot.y**2) * omega)
    return pot


@pytest.fixture
def factored():
    """A moving grid stored as an invariant lattice times a per-frame scale.

    Built exactly as a run saved with 'store_frame=False' is: the cartesian coordinates are
    dropped and the recipe for rebuilding them is left in an attribute instead, because
    materialising them would cost one full-size array per axis per frame.
    """
    pot = Potential(unitvecs=[[8, 0], [0, 10]], resolution=(NA1, NA2), v0=0)
    rho_x, rho_y = xr.broadcast(pot.x, pot.y)
    tv = np.linspace(0, 1, 4)
    t = xr.DataArray(tv, dims="t", coords={"t": tv})
    lam = 1 + 0.7 * t
    field = (np.exp(-(rho_x**2 + rho_y**2) / 4) * (1 + 0 * t)).transpose("t", "a1", "a2")
    out = field.rename("f").drop_vars(["x", "y"]).assign_coords(
        rho_x=rho_x.drop_vars(["x", "y"], errors="ignore"),
        rho_y=rho_y.drop_vars(["x", "y"], errors="ignore"),
        lambda_x=lam,
        lambda_y=lam * 1.1,
    )
    out.attrs["cart_factors"] = json.dumps(
        {"x": ["rho_x", "lambda_x"], "y": ["rho_y", "lambda_y"]}
    )
    return out


@pytest.fixture
def materialised(factored):
    """The same moving grid with its cartesian coordinates written out in full."""
    return factored.assign_coords(
        x=(factored.rho_x * factored.lambda_x).transpose("t", "a1", "a2"),
        y=(factored.rho_y * factored.lambda_y).transpose("t", "a1", "a2"),
    )


# --- helpers ----------------------------------------------------------------------------------

def draw_map(data, cart_axes, **kw):
    """Build a map, move it once, and hand back the artist's values and mesh."""
    fig, ax = plt.subplots()
    try:
        method = kw.pop("method", "pcolormesh")
        sliders, update, _ = create_map(fig, ax, cart_axes, data, method, **kw)
        values = {k: w.value for k, w in sliders.items()}
        update(**values)
        artist = ax.collections[-1]
        return (np.asarray(artist.get_array()).copy(),
                artist._coordinates.copy(), sliders, values)
    finally:
        plt.close(fig)


def draw_line(data, cart_axis, **kw):
    fig, ax = plt.subplots()
    try:
        sliders, update, _ = create_line(fig, ax, cart_axis, data, **kw)
        update(**{k: w.value for k, w in sliders.items()})
        line = ax.lines[-1]
        return line.get_xdata().copy(), line.get_ydata().copy(), sliders
    finally:
        plt.close(fig)


def draw_quiver(dataU, dataV, cart_axes, **kw):
    fig, ax = plt.subplots()
    try:
        sliders, update, _ = create_quiver(fig, ax, cart_axes, dataU, dataV, **kw)
        update(**{k: w.value for k, w in sliders.items()})
        return ax.collections[-1].get_offsets().copy(), sliders
    finally:
        plt.close(fig)


# --- the flat path: a static 2-D lattice drawn on its own dims ---------------------------------

def test_flat_mode_draws_the_field_in_its_own_dim_order(aligned2d):
    """The common case, and the one with no coverage until now.

    A static 2-D lattice at cart_axes [0,1] is drawn on a1,a2 with the dense cartesian
    coordinates as the mesh. The order the field is transposed into is read back off those
    coordinates, so it follows the array's own storage rather than a fixed (y, x) rule.
    """
    shown, mesh, sliders, _ = draw_map(aligned2d.V, [0, 1])
    assert np.allclose(shown.ravel(), aligned2d.V.transpose("a1", "a2").values.ravel())
    assert not np.allclose(shown.ravel(), aligned2d.V.transpose("a2", "a1").values.ravel())
    assert mesh.shape == (NA1 + 1, NA2 + 1, 2)
    assert sliders == {}


def test_flat_mode_picture_is_invariant_to_storage_order(aligned2d):
    """Storing the field as (a2, a1) must move the mesh with it, not flip the picture.

    The flattened artist array does change, because the field is drawn in whatever order it is
    stored. What must not change is where each value lands on the canvas.
    """
    straight, mesh_s, _, _ = draw_map(aligned2d.V, [0, 1])
    swapped, mesh_w, _, _ = draw_map(aligned2d.V.transpose("a2", "a1"), [0, 1])
    assert np.allclose(mesh_w, mesh_s.transpose(1, 0, 2))
    assert np.allclose(straight.reshape(NA1, NA2), swapped.reshape(NA2, NA1).T)


def test_flat_mode_does_not_interpolate_a_skewed_lattice(skewed2d, monkeypatch):
    """A skewed lattice at [0,1] keeps its skewed cells: interpolating would smooth them away."""
    called = []
    monkeypatch.setattr(plotting, "_to_orthogonal", lambda *a, **k: called.append(1) or a[0])
    _, mesh, _, _ = draw_map(skewed2d.V, [0, 1])
    assert called == []
    # a genuinely skewed quad mesh: the x edge coordinate varies down a column as well as across
    assert np.ptp(mesh[..., 0], axis=0).max() > 0


def test_flat_mode_keeps_parameter_dims_as_sliders(param2d):
    shown, _, sliders, values = draw_map(param2d.V, [0, 1])
    assert list(sliders) == ["omega"]
    assert values["omega"] == pytest.approx(1.0), "parameter sliders start at their left edge"
    truth = param2d.V.sel(omega=1.0, method="nearest").transpose("a1", "a2")
    assert np.allclose(shown.ravel(), truth.values.ravel())


# --- sliders ----------------------------------------------------------------------------------

def test_leftover_spatial_slider_starts_mid_and_parameters_start_left(aligned3d):
    """A leftover spatial axis starts in the middle, everything else at its left edge.

    After an orthogonal relabelling the edges of a spatial axis can fall outside the original
    lattice and read as all-NaN, so a spatial slider parked at its left edge would open on an
    empty panel.
    """
    stacked = aligned3d * (1 + 0 * create_parameter("g", np.linspace(2, 5, 3)))
    _, _, sliders, values = draw_map(stacked, [0, 2])
    assert list(sliders) == ["y", "g"], "leftover spatial axes come before parameters"
    ycoord = aligned3d.coords["y"]
    assert values["y"] == pytest.approx(
        (float(ycoord.min()) + float(ycoord.max())) / 2, abs=1e-9
    )
    assert values["g"] == pytest.approx(2.0)


# --- cart_axes given as dimension names --------------------------------------------------------

def test_string_cart_axes_draw_against_parameter_dims():
    """Two non-spatial dims can be plotted against each other, on data with no lattice at all.

    This is the ortho path with the relabelling degenerating to nothing, because '_aligned_axes'
    on an empty lattice returns an empty list rather than None. It has exactly one call site in
    the wild and had no coverage, so it is the path most likely to be lost in a refactor.
    """
    lam = np.linspace(0, 1, 5)
    dB = np.linspace(-1, 1, 7)
    data = xr.DataArray(
        np.arange(35.0).reshape(7, 5),
        dims=("dB0", "lambda"),
        coords={"dB0": dB, "lambda": lam},
        name="sig",
    )
    assert plotting._spatial_dims(data) == []
    assert plotting._aligned_axes(data, []) == []

    shown, mesh, sliders, _ = draw_map(data, ["lambda", "dB0"])
    assert sliders == {}
    assert np.allclose(shown.ravel(), data.transpose("dB0", "lambda").values.ravel())
    assert mesh.shape == (8, 6, 2)


# --- create_line -------------------------------------------------------------------------------

def test_line_on_a_1d_field_uses_the_lattice_itself():
    pot = Potential(unitvecs=[[8]], resolution=(20,), v0=0)
    pot.set(pot.x**2)
    x, y, sliders = draw_line(pot.V, 0)
    assert sliders == {}
    assert len(x) == 20
    assert np.allclose(np.asarray(y).ravel(), pot.V.values.ravel())


def test_line_through_a_2d_field_adds_a_leftover_slider(aligned2d):
    x, y, sliders = draw_line(aligned2d.V, 0)
    assert list(sliders) == ["y"]
    assert len(x) == NA1


def test_line_follows_a_materialised_moving_grid(materialised):
    """The abscissa moves with the frame, so the line is redrawn rather than updated in place."""
    fig, ax = plt.subplots()
    try:
        sliders, update, _ = create_line(fig, ax, 0, materialised)
        values = {k: w.value for k, w in sliders.items()}
        values["t"] = float(materialised.t[0])
        update(**values)
        first = ax.lines[-1].get_xdata().copy()
        values["t"] = float(materialised.t[-1])
        update(**values)
        last = ax.lines[-1].get_xdata().copy()
    finally:
        plt.close(fig)
    assert last.max() > first.max() * 1.5, "lambda grows by 1.7 over the run"


# --- create_quiver -----------------------------------------------------------------------------

def test_quiver_subsamples_by_density(aligned2d):
    """'density' thins the arrows, keeping every n-th point of the grid along each axis.

    The count is not asserted outright because these axes take the orthogonal path, so the grid
    is the interpolated one rather than the lattice. What matters is that thinning is a subset
    of the full set of positions, taken along both axes rather than one.
    """
    U = aligned2d.V.rename("U")
    V = (aligned2d.V * 0 + 1).rename("V")
    offsets, sliders = draw_quiver(U, V, [1, 0])
    assert sliders == {}

    dense = dict(quiver_tmpl())
    dense["density"] = 1
    offsets_all, _ = draw_quiver(U, V, [1, 0], template=dense)

    kept = {tuple(np.round(p, 9)) for p in offsets}
    every = {tuple(np.round(p, 9)) for p in offsets_all}
    assert kept < every, "thinned arrows must sit on positions the full set also has"
    ratio = len(every) / len(kept)
    assert 3.5 < ratio < 4.5, f"density 2 should thin both axes, got a factor {ratio:.2f}"


# --- bugs, recorded so that they cannot move unnoticed ------------------------------------------

def test_quiver_draws_on_its_default_axes(aligned2d):
    """create_quiver's own default used to raise, which made the common case unreachable.

    Both cartesian coordinates of a static lattice are dense over (a1, a2), so taking the first
    dim of each asked for the transposition ('a1', 'a1'). 'create_map' had the same rule written
    with a deduplication and was fine; sharing one layout is what makes the two agree.
    """
    U = aligned2d.V.rename("U")
    V = (aligned2d.V * 0 + 1).rename("V")
    offsets, sliders = draw_quiver(U, V, [0, 1])
    assert sliders == {}
    # drawn on the lattice itself, thinned by the default density of 2
    assert offsets.shape == (NA1 // 2 * (NA2 // 2), 2)


@pytest.mark.parametrize("helper", ["map", "line", "quiver"])
def test_every_helper_reads_a_factored_frame(factored, helper):
    """Only create_map used to pass the factors on to '_moving_dims'.

    Without them the field does not look like a moving grid at all, so the other two fell into
    the orthogonal path and went looking for cartesian coordinates that were deliberately never
    materialised. Every helper now recognises the recipe, so a rescaling run can be drawn as a
    line or a quiver and not only as a map.
    """
    fig, ax = plt.subplots()
    try:
        if helper == "map":
            sliders, update, _ = create_map(fig, ax, [0, 1], factored, "pcolormesh")
        elif helper == "line":
            sliders, update, _ = create_line(fig, ax, 0, factored)
        else:
            sliders, update, _ = create_quiver(fig, ax, [0, 1], factored, factored)
        assert "t" in sliders, "the frame parameter must drive a slider"
        update(**{k: w.value for k, w in sliders.items()})
        assert ax.collections or ax.lines
    finally:
        plt.close(fig)


def test_create_map_leaves_the_callers_template_alone(aligned2d):
    """The template a caller passes must come back untouched.

    'create_map' used to write its 'fkwargs' default into the dict before copying it, which bit
    hardest on the mutable default argument shared by every call that passed no template at all.
    """
    given = {}
    draw_map(aligned2d.V, [0, 1], template=given)
    assert given == {}

    styled = {"fkwargs": {"cmap": "viridis"}}
    draw_map(aligned2d.V, [0, 1], template=styled)
    assert styled == {"fkwargs": {"cmap": "viridis"}}


# --- templates ---------------------------------------------------------------------------------

def test_a_prefilled_template_survives_being_used(aligned2d):
    """A template built once and handed to several subplots must come back unchanged."""
    tmpl = cmesh_tmpl("amplitude")
    assert callable(tmpl["fkwargs"]["norm"])
    draw_map(aligned2d.V, [0, 1], template=tmpl)
    draw_map(aligned2d.V, [0, 1], template=tmpl)
    assert callable(tmpl["fkwargs"]["norm"]), "the norm factory was consumed in place"


def test_each_map_gets_its_own_norm(aligned2d):
    """Two maps sharing a template must not share a colour scale."""
    tmpl = cmesh_tmpl("amplitude")
    fig, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    try:
        create_map(fig, ax1, [0, 1], aligned2d.V, "pcolormesh", template=tmpl)
        create_map(fig2, ax2, [0, 1], aligned2d.V * 5, "pcolormesh", template=tmpl)
        assert ax1.collections[-1].norm is not ax2.collections[-1].norm
    finally:
        plt.close(fig)
        plt.close(fig2)


def test_clim_and_colorbar_labels_survive_a_redraw(aligned2d):
    """A contour is destroyed and rebuilt on every update, so its clim has to be reapplied."""
    tmpl = {"fkwargs": {"levels": 3}, "clim": (-1.0, 2.0)}
    fig, ax = plt.subplots()
    try:
        sliders, update, _ = create_map(fig, ax, [0, 1], aligned2d.V, "contour", template=tmpl)
        assert ax.collections[-1].get_clim() == (-1.0, 2.0)
        update(**{k: w.value for k, w in sliders.items()})
        assert ax.collections[-1].get_clim() == (-1.0, 2.0)
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        create_map(fig, ax, [0, 1], aligned2d.V, "pcolormesh", template=cmesh_tmpl("phase"))
        assert ax.collections[-1].get_clim() == pytest.approx((-np.pi, np.pi))
        labels = [t.get_text() for t in fig.axes[-1].get_yticklabels()]
        assert labels == [r"$-\pi$", "0", r"$\pi$"]
    finally:
        plt.close(fig)


# --- the high-level entry points ----------------------------------------------------------------

def test_plot_eigenvector_mixes_axes_and_shares_sliders(param2d):
    """One slider drives every subplot that names the same dimension.

    The second cell asks for a single cartesian axis, so it is a line rather than a map, and the
    two cells still have to agree on 'omega'.
    """
    fig, axes = plot_eigenvector(
        [[param2d.V, param2d.V]],
        [[param2d, None]],
        [["amplitude", None]],
        cart_axes=[[[0, 1], [0]]],
    )
    try:
        assert np.shape(axes) == (1, 2)
        assert axes[0][0].collections, "cell 0 draws a mesh"
        assert axes[0][1].lines, "cell 1 draws a line"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "template",
    [
        None,
        "amplitude",
        {"fkwargs": {"cmap": "viridis"}},
        ("amplitude",),
        ("amplitude", contour_tmpl(2)),
        ("amplitude", contour_tmpl(2), quiver_tmpl()),
    ],
    ids=["none", "string", "dict", "tuple1", "tuple2", "tuple3"],
)
def test_plot_eigenvector_accepts_every_template_shape(param2d, template):
    fig, axes = plot_eigenvector([[param2d.V]], [[param2d]], [[template]])
    try:
        assert axes[0][0].collections
    finally:
        plt.close(fig)


def test_dashboard_draws_bands_and_maps(param2d):
    band = xr.DataArray(
        np.linspace(0, 1, 3 * 2).reshape(3, 2),
        dims=("omega", "band"),
        coords={"omega": param2d.V.coords["omega"], "band": [0, 1]},
        name="E",
    )
    fig, ax = dashboard(band, "omega", [[param2d.V]], param2d, "amplitude")
    try:
        assert ax.lines, "the band panel draws one line per band"
    finally:
        plt.close(fig)


def test_dashboard_styles_its_maps(param2d):
    """The dashboard used to pass its template into the 'resolution' slot, so nothing was styled.

    Both of its 'create_map' calls were positional, and a 'resolution' parameter had since been
    inserted ahead of 'template'. The template went to the wrong place, the real one stayed
    empty, and every panel drew in matplotlib's default colours instead of the requested ones.
    """
    band = xr.DataArray(
        np.linspace(0, 1, 6).reshape(3, 2),
        dims=("omega", "band"),
        coords={"omega": param2d.V.coords["omega"], "band": [0, 1]},
        name="E",
    )
    fig, _ = dashboard(band, "omega", [[param2d.V]], param2d, "amplitude")
    try:
        meshes = [c for a in fig.axes for c in a.collections if hasattr(c, "get_cmap")]
        assert meshes, "the eigenvector panel draws a mesh"
        assert any(m.get_cmap().name == cmesh_tmpl("amplitude")["fkwargs"]["cmap"].name
                   for m in meshes), "the 'amplitude' colormap never reached the mesh"
    finally:
        plt.close(fig)


def test_plot_eigenvector_overlays_a_quiver(param2d):
    """The three-layer template shape, which only one notebook exercises.

    The quiver's template is the third layer, and it reaches 'create_quiver' past two arguments
    that were added later. Nothing caught it when that call went stale.
    """
    U = param2d.V.rename("U")
    V = (param2d.V * 0 + 1).rename("V")
    fig, axes = plot_eigenvector(
        [[param2d.V]],
        [[param2d]],
        [[("amplitude", contour_tmpl(2), quiver_tmpl())]],
        quivers=[[(U, V)]],
    )
    try:
        from matplotlib.quiver import Quiver

        assert any(isinstance(c, Quiver) for c in axes[0][0].collections), "no arrows drawn"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "call",
    [
        lambda fig, ax, d: create_map(fig, ax, [0, 1], d, "pcolormesh", {"fkwargs": {}}),
        lambda fig, ax, d: create_line(fig, ax, 0, d, {"fkwargs": {}}),
        lambda fig, ax, d: create_quiver(fig, ax, [0, 1], d, d, {"fkwargs": {}}),
    ],
    ids=["map", "line", "quiver"],
)
def test_options_must_be_passed_by_keyword(aligned2d, call):
    """Passing a template positionally is refused rather than silently misread.

    The three helpers had grown different orders for the same four options, and a template
    landing in the 'resolution' slot disabled styling without any complaint. Keyword-only
    arguments turn that into an error at the call site.
    """
    fig, ax = plt.subplots()
    try:
        with pytest.raises(TypeError, match="positional"):
            call(fig, ax, aligned2d.V)
    finally:
        plt.close(fig)


def test_cart_extent_matches_the_materialised_product(factored, materialised):
    """The span of a factored coordinate is taken from its factors, never by multiplying them out.

    Building the product to measure it would cost exactly the array the factored form exists to
    avoid, so the corners of the factors are used instead. That shortcut is only worth anything
    if it gives the same answer, which is what this checks.
    """
    for name in ("x", "y"):
        lean = plotting._cart_extent(factored, name, FACTORS)
        full = plotting._cart_extent(materialised, name, {})
        assert lean == pytest.approx(full)


@pytest.mark.parametrize("field", ["factored", "materialised"])
def test_cst_bds_pins_the_axes_to_the_largest_frame(request, field):
    """A moving grid held at constant bounds must span every frame, not just the first.

    This used to raise outright on a factored field: the check for whether the grid moves was
    made without the recipe for rebuilding its coordinates, so it went looking for a coordinate
    that is deliberately not stored. Both storage forms have to give the same box.
    """
    data = request.getfixturevalue(field)
    fig, axes = plot_eigenvector([[abs(data)]], [[None]], [["amplitude"]], cst_bds=True)
    try:
        rho = data.coords["rho_x"]
        lam = data.coords["lambda_x"]
        widest = float(lam.max())
        assert axes[0][0].get_xlim() == pytest.approx(
            (float(rho.min()) * widest, float(rho.max()) * widest)
        )
    finally:
        plt.close(fig)
