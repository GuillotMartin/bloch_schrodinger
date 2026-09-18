"""What the preset templates promise, including the two properties that make them correct.

Most of these are ordinary checks that a preset draws and that its options land where they say.
Two are different in kind, and are the reason this file exists rather than a handful of extra
cases elsewhere: a diverging preset must have a neutral midpoint, and the phase preset's map
must close. Both were broken, both are easy to break again by swapping a colormap that looks
nice, and neither shows up as an error -- only as a picture that says the wrong thing.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from bloch_schrodinger.plotting import (
    LayerPair,
    cmesh_tmpl,
    contour_tmpl,
    create_map,
    plot_eigenvector,
    signed_log_tmpl,
)
from bloch_schrodinger.potential import Potential

PRESETS = [
    "amplitude",
    "amplitude - log",
    "real",
    "real - log",
    "phase",
    "potential",
    "difference",
    "spin",
]

# The presets whose job is to show a signed quantity either side of zero.
DIVERGING = ["real", "real - log", "potential", "difference", "spin"]


@pytest.fixture
def field():
    """A signed 2-D field, non-square, spanning several decades so a log scale has work to do."""
    pot = Potential(unitvecs=[[8, 0], [0, 10]], resolution=(12, 14), v0=0)
    pot.set(np.exp(-(pot.x**2 + pot.y**2) / 4) * pot.x)
    return pot.V


def resolve(cmap):
    """A template's cmap entry, as a Colormap, whether it was given by name or by object."""
    return matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap


# --- the two properties that make a preset correct ---------------------------------------------

@pytest.mark.parametrize("name", DIVERGING)
def test_diverging_presets_have_a_neutral_midpoint(name):
    """Zero must read as nothing, which means the middle of the map is a light neutral.

    'real' used to default to 'berlin', whose midpoint is near-black, so the one value a signed
    field is centred on came out looking like an extreme. The test is on the colormap rather
    than on a name, so swapping in another map cannot reintroduce the problem quietly.
    """
    mid = np.asarray(resolve(cmesh_tmpl(name)["fkwargs"]["cmap"])(0.5)[:3])
    assert mid.max() - mid.min() < 0.10, f"{name}: midpoint is a hue, not a neutral ({mid})"
    assert mid.mean() > 0.75, f"{name}: midpoint is dark ({mid.mean():.3f}), so zero reads as extreme"


def test_phase_preset_uses_a_colormap_that_closes():
    """A wrapped quantity needs a cyclic map, or the jump from +pi to -pi draws as a seam."""
    cmap = resolve(cmesh_tmpl("phase")["fkwargs"]["cmap"])
    ends = np.abs(np.asarray(cmap(0.0)[:3]) - np.asarray(cmap(1.0)[:3]))
    assert ends.max() < 0.05, f"the two ends of the phase colormap do not meet (gap {ends.max():.3f})"


def test_phase_preset_spans_exactly_one_period():
    """Anything less and the wrap still shows, because the two ends stop being the same value."""
    fkwargs = cmesh_tmpl("phase")["fkwargs"]
    assert (fkwargs["vmin"], fkwargs["vmax"]) == pytest.approx((-np.pi, np.pi))


# --- every preset is usable --------------------------------------------------------------------

@pytest.mark.parametrize("name", PRESETS)
def test_every_preset_draws(field, name):
    """Checked by drawing, because a template is only correct against a real matplotlib call."""
    fig, ax = plt.subplots()
    try:
        sliders, update, _ = create_map(
            fig, ax, [0, 1], field, "pcolormesh", template=cmesh_tmpl(name)
        )
        update(**{k: w.value for k, w in sliders.items()})
        assert ax.collections
    finally:
        plt.close(fig)


def test_an_unknown_preset_name_is_refused():
    """It used to return None, which drew unstyled and so made a mistyped name invisible."""
    with pytest.raises(ValueError, match="unknown template") as excinfo:
        cmesh_tmpl("dnesity")
    assert "amplitude" in str(excinfo.value), "the error should list the names that do work"


@pytest.mark.parametrize("name", PRESETS)
def test_presets_share_no_state_between_calls(name):
    """A template is handed to the caller and often patched, so two calls must be independent."""
    first, second = cmesh_tmpl(name), cmesh_tmpl(name)
    assert first is not second
    assert first["fkwargs"] is not second["fkwargs"]
    first["fkwargs"]["cmap"] = "nonsense"
    first.setdefault("colorbar", {}).setdefault("kwargs", {})["label"] = "nonsense"
    assert second["fkwargs"]["cmap"] != "nonsense"
    assert second.get("colorbar", {}).get("kwargs", {}).get("label") != "nonsense"


# --- the keyword overrides ---------------------------------------------------------------------

def test_overrides_land_where_they_say():
    temp = cmesh_tmpl(
        "amplitude",
        cmap="magma",
        label=r"$|\psi|^2$",
        fmt="{x:.2f}",
        ticks=[-1, 1],
        tickslabel=[r"$\sigma_-$", r"$\sigma_+$"],
    )
    assert temp["fkwargs"]["cmap"] == "magma"
    assert temp["colorbar"]["kwargs"]["label"] == r"$|\psi|^2$"
    assert temp["colorbar"]["kwargs"]["format"] == "{x:.2f}"
    assert temp["colorbar"]["ticks"] == [-1, 1]
    assert temp["colorbar"]["tickslabel"] == [r"$\sigma_-$", r"$\sigma_+$"]


def test_fixed_limits_turn_autoscaling_off():
    """The two always went together by hand: limits that get rescaled away are no limits at all."""
    assert cmesh_tmpl("amplitude - log")["autoscale"] is True
    temp = cmesh_tmpl("amplitude - log", clim=(1e-12, 1e-1))
    assert temp["clim"] == (1e-12, 1e-1)
    assert temp["autoscale"] is False


def test_a_name_only_call_is_unchanged_by_the_new_options(field):
    """Every existing notebook calls these with a name alone, and must get what it always got."""
    bare = cmesh_tmpl("real")
    assert bare["autoscale"] is True
    assert "clim" not in bare
    assert set(bare["colorbar"]["kwargs"]) == {"format"}


def test_fixed_limits_reach_the_artist(field):
    fig, ax = plt.subplots()
    try:
        create_map(
            fig, ax, [0, 1], field, "pcolormesh",
            template=cmesh_tmpl("real", clim=(-2.0, 2.0)),
        )
        assert ax.collections[-1].get_clim() == (-2.0, 2.0)
    finally:
        plt.close(fig)


# --- the signed-log pair -----------------------------------------------------------------------

@pytest.mark.parametrize("decades", [3, 6, 10])
def test_signed_log_bands_colours_and_labels_line_up(decades):
    """One colour per band, one label per boundary, and the band across zero left blank.

    The hand-written version passed one colour too many and relied on matplotlib dropping the
    last, which works but hides an off-by-one if the level arithmetic ever changes.
    """
    filled, outline = signed_log_tmpl(decades)
    levels = filled["fkwargs"]["levels"]
    band_colors = filled["fkwargs"]["colors"]

    assert len(levels) == 2 * decades
    assert len(band_colors) == len(levels) - 1, "one colour per band, not per boundary"
    assert len(filled["colorbar"]["tickslabel"]) == len(levels)

    zero_band = int(np.searchsorted(levels, 0.0)) - 1
    assert np.allclose(band_colors[zero_band][:3], 1.0), "the band across zero must read as empty"
    assert outline["fkwargs"]["levels"] is levels, "outlines must sit on the band edges"


def test_signed_log_levels_are_symmetric_and_end_at_one():
    filled, _ = signed_log_tmpl(6)
    levels = filled["fkwargs"]["levels"]
    assert np.allclose(levels, -levels[::-1]), "the ladder must be symmetric about zero"
    assert levels[0] == -1.0 and levels[-1] == 1.0
    labels = filled["colorbar"]["tickslabel"]
    assert (labels[0], labels[-1]) == ("$-1$", "$1$")


def test_signed_log_reproduces_the_hand_written_version():
    """The five notebooks that build this by hand must keep getting the same picture.

    This is the one case where 'the same as before' is checkable exactly rather than by eye, so
    it is checked exactly, against the code as written in docs/Wannier.ipynb cell 11.
    """
    from cmcrameri.cm import vik

    nlevels = 6
    levels = np.logspace(-nlevels + 1, 0, nlevels)
    levels = np.append(-levels[::-1], levels)
    by_hand = vik(np.linspace(0, 1, nlevels * 2 + 1, endpoint=True)[1:])
    by_hand[nlevels - 1] = [1, 1, 1, 1]

    x = np.linspace(-1, 1, 40)
    X, Y = np.meshgrid(x, x)
    Z = np.sign(X) * np.abs(X) ** 2

    drawn = []
    for cols in (by_hand, signed_log_tmpl(nlevels).filled["fkwargs"]["colors"]):
        fig, ax = plt.subplots()
        try:
            cs = ax.contourf(X, Y, Z, levels=levels, colors=cols)
            drawn.append(np.asarray(cs.get_facecolor()).copy())
        finally:
            plt.close(fig)

    assert np.allclose(drawn[0], drawn[1]), "the preset draws different colours than the notebook"


def test_signed_log_pair_is_not_mistaken_for_a_template_entry(field):
    """Both layers describe the same field, so it is not (field, potential-contour).

    A bare tuple would be read as the latter and would quietly style the potential's contours
    with the outline, which is a wrong picture rather than an error.
    """
    pair = signed_log_tmpl(6)
    assert not isinstance(pair, tuple)
    assert isinstance(pair, LayerPair)
    with pytest.raises(ValueError, match="two passes over the same field"):
        plot_eigenvector([[field]], [[None]], [[pair]])


def test_signed_log_draws_as_the_two_passes_it_describes(field):
    """The way it is actually used: contourf for the bands, contour for their outlines."""
    normalised = field / abs(field).max()
    filled, outline = signed_log_tmpl(6, label=r"$\mathrm{sign}(w)|w|^2$")
    fig, ax = plt.subplots()
    try:
        s1, u1, ax = create_map(fig, ax, [0, 1], normalised, "contourf", template=filled)
        s2, u2, ax = create_map(fig, ax, [0, 1], normalised, "contour", template=outline)
        u1(**{k: w.value for k, w in s1.items()})
        u2(**{k: w.value for k, w in s2.items()})
        assert ax.collections
    finally:
        plt.close(fig)


# --- the axes section --------------------------------------------------------------------------

def test_axes_section_sets_labels_and_aspect(field):
    """Absorbs the set_xlabel / set_ylabel / set_aspect trio written out after every plot call."""
    temp = cmesh_tmpl("amplitude")
    temp["axes"] = {"xlabel": r"$x$ ($\mu$m)", "ylabel": r"$y$ ($\mu$m)", "aspect": "equal"}
    fig, ax = plt.subplots()
    try:
        create_map(fig, ax, [0, 1], field, "pcolormesh", template=temp)
        assert ax.get_xlabel() == r"$x$ ($\mu$m)"
        assert ax.get_ylabel() == r"$y$ ($\mu$m)"
        assert ax.get_aspect() == 1.0
    finally:
        plt.close(fig)


def test_plot_eigenvector_does_not_overwrite_a_requested_aspect(field):
    """It forces 'equal' on two-axis cells, which a template must still be able to override."""
    temp = cmesh_tmpl("amplitude")
    temp["axes"] = {"aspect": "auto", "xlabel": "not equal"}
    fig, axes = plot_eigenvector([[field]], [[None]], [[temp]])
    try:
        assert axes[0][0].get_aspect() == "auto"
        assert axes[0][0].get_xlabel() == "not equal"
    finally:
        plt.close(fig)

    plain, plain_axes = plot_eigenvector([[field]], [[None]], [["amplitude"]])
    try:
        assert plain_axes[0][0].get_aspect() == 1.0, "the default is still equal"
    finally:
        plt.close(plain)


def test_contour_tmpl_takes_its_style_as_arguments():
    """The levels, colour and linewidth were patched by hand in four notebooks."""
    temp = contour_tmpl([5], colors="k", linewidths=0.2, linestyles="solid")
    assert temp["fkwargs"] == {
        "levels": [5],
        "colors": "k",
        "linewidths": 0.2,
        "linestyles": "solid",
    }


def test_new_presets_carry_the_labels_they_were_added_for():
    assert cmesh_tmpl("potential")["colorbar"]["kwargs"]["label"] == r"$V/E_r$"
    assert cmesh_tmpl("spin")["colorbar"]["tickslabel"] == [
        r"$\sigma_-$", r"$\pi$", r"$\sigma_+$",
    ]
    assert cmesh_tmpl("difference")["colorbar"]["kwargs"]["format"].startswith("{x:+")


@pytest.mark.parametrize(
    "norm",
    [
        pytest.param(lambda: matplotlib.colors.LogNorm(), id="factory"),
        pytest.param(matplotlib.colors.LogNorm(), id="instance"),
    ],
)
def test_a_norm_may_be_given_as_a_factory_or_an_instance(field, norm):
    """Writing colors.LogNorm() is the natural thing, and used to raise from inside matplotlib.

    A Normalize is itself callable, since it maps values into zero-to-one, so an instance was
    indistinguishable from a factory and got called with no arguments. The resulting TypeError
    named only matplotlib internals, which points a template author nowhere.
    """
    positive = abs(field) + 1e-6
    fig, ax = plt.subplots()
    try:
        sliders, update, _ = create_map(
            fig, ax, [0, 1], positive, "pcolormesh", template={"fkwargs": {"norm": norm}}
        )
        update(**{k: w.value for k, w in sliders.items()})
        assert isinstance(ax.collections[-1].norm, matplotlib.colors.LogNorm)
    finally:
        plt.close(fig)


def test_two_maps_never_share_a_colour_scale(field):
    """Whichever form the norm was given in, autoscaling one map must not rescale another."""
    positive = abs(field) + 1e-6
    shared_instance = matplotlib.colors.Normalize()
    for norm in (lambda: matplotlib.colors.Normalize(), shared_instance):
        figs, norms = [], []
        try:
            for scale in (1, 100):
                fig, ax = plt.subplots()
                figs.append(fig)
                create_map(
                    fig, ax, [0, 1], positive * scale, "pcolormesh",
                    template={"fkwargs": {"norm": norm}, "autoscale": True},
                )
                norms.append(ax.collections[-1].norm)
            assert norms[0] is not norms[1]
            assert norms[0] is not shared_instance
        finally:
            for fig in figs:
                plt.close(fig)
