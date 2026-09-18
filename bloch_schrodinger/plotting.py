import json
import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from types import NoneType

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from cmcrameri import cm
from IPython.display import display
from ipywidgets import FloatSlider, HBox, VBox, interactive_output
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import marching_cubes

from bloch_schrodinger.potential import Potential
from bloch_schrodinger.utils import (
    create_cart_grid,
    create_sliders,
    create_sliders_from_dims,
)

coord_names = ["x", "y", "z"]


def _spatial_dims(data: xr.DataArray) -> list[str]:
    """Return the data's own lattice dims (a1, a2, ...), sorted by axis index."""
    return sorted(
        (d for d in data.dims if d[0] == "a" and d[1:].isdigit()),
        key=lambda d: int(d[1:]),
    )


def _to_orthogonal(
    data: xr.DataArray, spatial_dims: list[str], resolution: int|tuple[int] = None
) -> xr.DataArray:
    """Interpolate data onto an orthogonal cartesian grid, replacing its a1,a2,... dims with x,y,z.
    Used whenever the axes to plot against aren't the array's own native a1,a2 axes."""
    n_dims = len(spatial_dims)
    if resolution is None:
        resolution = max(data.sizes[d] for d in spatial_dims)
    inv_coords = create_cart_grid(data, resolution=resolution)
    mapping = {f"a{i + 1}": inv_coords[i] for i in range(n_dims)}
    return (
        data.interp(mapping, method="linear")
        .drop_vars([coord_names[i] for i in range(n_dims)])
        .rename({coord_names[i] + "n": coord_names[i] for i in range(n_dims)})
    )


def _rank1_axis(values: np.ndarray) -> int | None:
    """The single array axis a coordinate actually varies along, or None if it varies along more.

    A cartesian coordinate built over a lattice is stored dense, with one entry per lattice point,
    even when the box is axis-aligned and the coordinate really only depends on one of them. This
    is the same test 'Potential.coords_1d' makes on the other side of the package: an axis-aligned
    coordinate has zero peak-to-peak spread along every axis but its own.

    Args:
        values (np.ndarray): A coordinate evaluated over the lattice.

    Returns:
        int or None: The axis it varies along, or None if no single axis accounts for it.
    """
    for i in range(values.ndim):
        others = tuple(j for j in range(values.ndim) if j != i)
        if not others or np.ptp(values, axis=others).max() == 0:
            return i
    return None


def _aligned_axes(data: xr.DataArray, spatial_dims: list[str]) -> list[int] | None:
    """Which lattice axis each cartesian coordinate runs along, or None for a skewed lattice.

    When every cartesian coordinate varies along exactly one lattice axis -- and no two share one --
    the lattice already *is* a cartesian grid, just labelled by lattice index rather than by
    position. Plotting it then needs no interpolation at all: the data can be drawn on its own
    lattice and the axes labelled from the coordinates, which is what the moving-grid path does.

    This matters because the alternative is expensive out of all proportion. '_to_orthogonal'
    interpolates the *whole* array at setup, and on a swept 3-D run that is tens of gigabytes of
    'interp' to achieve what is, in this case, a transposition and a relabelling.

    A genuinely skewed lattice -- a triangular one, say -- returns None and is interpolated as
    before, since there the cartesian and lattice axes really do mix.

    Args:
        data (xr.DataArray): The field, carrying cartesian coordinates over its lattice.
        spatial_dims (list[str]): The lattice dims, in axis order.

    Returns:
        list[int] or None: For each cartesian axis, the index into 'spatial_dims' it runs along;
        None if the lattice is skewed, or if a cartesian coordinate is missing or depends on a
        non-lattice dimension.
    """
    axes = []
    for i in range(len(spatial_dims)):
        name = coord_names[i]
        if name not in data.coords:
            return None  # absent, or stored factored: either way not a static aligned lattice
        coord = data.coords[name]
        own = [d for d in spatial_dims if d in coord.dims]
        if set(coord.dims) - set(spatial_dims) or not own:
            return None  # varies with a parameter: a moving grid, handled elsewhere
        axis = _rank1_axis(np.asarray(coord.transpose(*own).data))
        if axis is None:
            return None
        axes.append(spatial_dims.index(own[axis]))
    return axes if len(set(axes)) == len(axes) else None


def _cart_factors(data: xr.DataArray, frame: dict | None = None) -> dict:
    """The recipe for any cartesian coordinate stored as a product of other coordinates.

    A moving frame is usually an outer product: a rescaling solver's grid is x = rho_x * lambda(t),
    an invariant lattice times one number per time step. Stored as x itself that is an array the
    size of the field; stored as its two factors it is a few kilobytes. This lets an array say so,
    so the plotter can rebuild one frame at a time instead of being handed the whole product.

    Nothing here knows what the factors mean or what they are called -- only that a coordinate may
    be the product of others on the same array. The recipe comes from the explicit 'frame' argument
    if given, else from the array's own 'cart_factors' attribute, which is JSON because netCDF
    attributes cannot hold a mapping.

    Args:
        data (xr.DataArray): The field.
        frame (dict, optional): An explicit recipe, e.g. {"x": ("rho_x", "lambda_x")}, overriding
        whatever the array carries. Defaults to None.

    Returns:
        dict: Cartesian coordinate name -> the names of its factors. Empty when there are none.
    """
    if frame is not None:
        return frame
    raw = data.attrs.get("cart_factors")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw or {}


def _cart_dims(data: xr.DataArray, name: str, factors: dict) -> tuple[str, ...]:
    """The dims a cartesian coordinate depends on, whether it is stored or factored."""
    if name in data.coords:
        return tuple(data.coords[name].dims)
    return tuple(
        dict.fromkeys(d for part in factors[name] for d in data.coords[part].dims)
    )


def _cart_at(data: xr.DataArray, name: str, factors: dict, sel: dict) -> xr.DataArray:
    """One frame of a cartesian coordinate, rebuilt from its factors when it is not stored.

    Each factor is selected down to this frame *before* the product is taken. Multiplying first
    would build the whole outer product -- the very array the factored form exists to avoid -- and
    only then throw all but one frame of it away.

    Args:
        data (xr.DataArray): The field.
        name (str): The cartesian coordinate wanted.
        factors (dict): From '_cart_factors'.
        sel (dict): The current slider position.

    Returns:
        xr.DataArray: The coordinate over the plotted lattice, for this frame.
    """
    def at(coord):
        return coord.sel(
            {d: v for d, v in sel.items() if d in coord.dims}, method="nearest"
        )

    if name in data.coords:
        return at(data.coords[name])
    out = None
    for part in factors[name]:
        piece = at(data.coords[part])
        out = piece if out is None else out * piece
    return out


def _axis_names(cart_axes: list) -> list[str]:
    """The coordinate name each entry of 'cart_axes' refers to.

    An entry is either a cartesian axis index (0 = 'x', 1 = 'y', 2 = 'z') or, to plot a field
    against its parameters rather than against space, the name of a dimension outright.
    """
    return [coord_names[c] if isinstance(c, int) else c for c in cart_axes]


def _cart_extent(data: xr.DataArray, name: str, factors: dict) -> tuple[float, float]:
    """The full span of a cartesian coordinate, over every frame the run reaches.

    This is the box to pin the axes to when a moving grid is meant to be seen growing rather
    than rescaled away. Stored dense, the span is simply the coordinate's own min and max.
    Stored as a product of factors, taking it that way would mean building the very outer
    product the factored form exists to avoid, so it is taken from the factors' corners
    instead: the extremes of a product of independent ranges are always attained at some
    combination of the ranges' own extremes.

    Args:
        data (xr.DataArray): The field.
        name (str): The cartesian coordinate wanted.
        factors (dict): From '_cart_factors'.

    Returns:
        tuple[float, float]: The lowest and highest value the coordinate takes anywhere.
    """
    if name in data.coords:
        coord = data.coords[name]
        return float(coord.min()), float(coord.max())

    lo = hi = None
    for part in factors[name]:
        part_coord = data.coords[part]
        ends = (float(part_coord.min()), float(part_coord.max()))
        if lo is None:
            lo, hi = ends
        else:
            products = [a * b for a in (lo, hi) for b in ends]
            lo, hi = min(products), max(products)
    return lo, hi


def _relabel_cartesian(
    data: xr.DataArray, spatial_dims: list[str], aligned: list[int]
) -> xr.DataArray:
    """Replace the lattice dims with the cartesian coordinates they run along.

    This is '_to_orthogonal' for a lattice that is already cartesian: the same relabelling of
    a1, a2, ... into x, y, ..., but taken exactly rather than interpolated. Every cartesian
    coordinate here varies along a single lattice axis, so its dense form is a broadcast copy of
    one line of values, and that line can simply become the axis' index.

    Doing it this way is what keeps the rest of the function -- the sliders, their names and their
    physical values, the transposes -- working exactly as it did when the data really was
    interpolated. Only the cost changes: 'interp' over the whole array becomes a reshape of a few
    hundred numbers.

    Args:
        data (xr.DataArray): The field, on an axis-aligned lattice.
        spatial_dims (list[str]): The lattice dims, in axis order.
        aligned (list[int]): From '_aligned_axes': the lattice axis each cartesian axis runs along.

    Returns:
        xr.DataArray: The same data, indexed by x, y, ... instead of a1, a2, ...
    """
    out = data
    for i, lattice_axis in enumerate(aligned):
        name, dim = coord_names[i], spatial_dims[lattice_axis]
        line = out.coords[name]
        line = line.isel({d: 0 for d in line.dims if d != dim})
        out = out.assign_coords({name: (dim, np.asarray(line.data))}).swap_dims({dim: name})
    return out.drop_vars(
        [d for d in spatial_dims if d in out.coords], errors="ignore"
    )


def _moving_dims(
    data: xr.DataArray, name: str, spatial_dims: list[str], factors: dict | None = None
) -> list[str]:
    """Return the non-spatial dims the cartesian coordinate 'name' depends on.

    A field living on a moving grid -- the output of a rescaling solver, say -- carries cartesian
    coordinates that are functions of the parameters, x(t, ...), rather than of the lattice alone.
    An empty list means the grid is fixed and the coordinate can be read once and reused."""
    return [
        d
        for d in _cart_dims(data, name, _cart_factors(data, factors))
        if d not in spatial_dims
    ]


def _make_sliders(
    data: xr.DataArray,
    slider_dims: list[str],
    leftover_spatial: list[str],
    template: dict,
) -> dict:
    """Build sliders for slider_dims. Leftover spatial axes (e.g. "z" when plotting x,y out of a 3D field)
    start at 'mid' rather than the template's requested start, since after orthogonal interpolation the
    edges of a spatial axis can be entirely outside the original lattice and thus all-NaN."""
    spatial = [d for d in slider_dims if d in leftover_spatial]
    other = [d for d in slider_dims if d not in leftover_spatial]
    return {
        **create_sliders_from_dims({d: data.coords[d] for d in spatial}, start="mid"),
        **create_sliders_from_dims(
            {d: data.coords[d] for d in other},
            start=template.get("slider_start", "left"),
        ),
    }



@dataclass(eq=False)
class _Drawable:
    """Everything the drawing helpers need to know about how a field maps onto a pair of axes.

    'eq=False' matters: the generated __eq__ would compare DataArray fields elementwise and
    return an array, which is not something a dataclass can use as a truth value.

    Attributes:
        data (xr.DataArray): The field, relabelled or interpolated onto cartesian axes where the
            mode calls for it, and otherwise exactly as it was passed in.
        mode (str): 'moving', 'ortho' or 'flat'. See '_prepare' for what each one means.
        axes (list[str]): The coordinate names plotted against, in the order 'cart_axes' asked
            for. Used for the axis labels and to look the mesh up.
        plotted_dims (list[str]): The dims of 'data' that are drawn rather than sliced.
        sliders (dict): One widget per dim that is not drawn.
        mesh_at (Callable): sel -> one coordinate per plotted axis, at that slider position.
        field_at (Callable): (array, sel) -> that array sliced to the frame and transposed to
            match the mesh. It takes the array as an argument rather than closing over it, so
            that a second field on the same lattice -- a quiver's V to its U -- can be pushed
            through the same layout.
        align (Callable): array -> the same relabelling or interpolation this layout applied to
            'data', for a second field that has to end up on the same axes.
    """

    data: xr.DataArray
    mode: str
    axes: list[str]
    plotted_dims: list[str]
    sliders: dict
    mesh_at: Callable
    field_at: Callable
    align: Callable


def _prepare(
    data: xr.DataArray,
    cart_axes: list,
    template: dict | None = None,
    resolution: int | tuple[int] | None = None,
    frame: dict | None = None,
) -> _Drawable:
    """Work out how a field should be laid out on the given axes, without drawing anything.

    This is the part 'create_map', 'create_line' and 'create_quiver' used to each carry their
    own copy of, and the part they kept drifting apart on. There are three modes:

    'moving' -- the cartesian coordinates depend on the parameters, as a rescaling solver's do,
    so there is no single grid to interpolate onto: the field is drawn on its own lattice and
    the mesh is rebuilt at each slider position. This also draws a skewed lattice as the skewed
    cells it really is rather than smoothing over them.

    'ortho' -- the axes asked for are not the field's own first two lattice axes, so it has to
    be re-indexed by cartesian coordinate. An axis-aligned lattice is already a cartesian grid
    wearing lattice labels, so that is an exact relabelling; only a genuinely skewed one, where
    the cartesian and lattice axes really do mix, is worth interpolating. The difference is not
    cosmetic: interpolating a swept 3-D run means tens of gigabytes of work to achieve what is
    otherwise a transposition.

    'flat' -- a static field drawn against its own leading lattice axes. The dense cartesian
    coordinates are handed to matplotlib as the mesh, so a skewed lattice keeps its true shape
    here too, and nothing is interpolated.

    Nothing in here touches the field's values. Everything that would is deferred into the two
    callables, so that one frame of a lazily-chunked sweep costs one frame's worth of work.

    Args:
        data (xr.DataArray): The field to lay out.
        cart_axes (list): One entry per axis to draw against, either a cartesian axis index
            (0 = 'x', 1 = 'y', 2 = 'z') or the name of a non-spatial dimension. One entry gives
            a line, two give a map.
        template (dict, optional): Consulted only for 'slider_start'. Defaults to None.
        resolution (int or tuple[int], optional): The resolution to interpolate onto, when a
            skewed lattice has to be. The field's own is used if None. Defaults to None.
        frame (dict, optional): An explicit recipe for coordinates stored as a product of
            others, overriding the array's own 'cart_factors'. Defaults to None.

    Returns:
        _Drawable: The layout.
    """
    spatial_dims = _spatial_dims(data)
    n_dims = len(spatial_dims)
    n_axes = len(cart_axes)

    # An entry may name a non-spatial dimension instead of a cartesian axis, which is how a
    # field is plotted against two parameters rather than against space.
    spatial_plot = any(isinstance(c, int) for c in cart_axes)
    axes = _axis_names(cart_axes)

    factors = _cart_factors(data, frame)
    moving = bool(_moving_dims(data, axes[0], spatial_dims, factors)) and bool(spatial_dims)
    ortho = not moving and (list(cart_axes) != list(range(n_axes)) or n_dims != n_axes)

    def align(arr: xr.DataArray) -> xr.DataArray:
        return arr

    if ortho:
        aligned = _aligned_axes(data, spatial_dims)
        if aligned is None:
            def align(arr):
                return _to_orthogonal(arr, _spatial_dims(arr), resolution)
        else:
            def align(arr):
                return _relabel_cartesian(arr, _spatial_dims(arr), aligned)
        data = align(data)

    if not spatial_plot:
        plotted_dims = list(axes)
        leftover_spatial = [coord_names[d] for d in range(n_dims)] if ortho else []
    elif moving:
        # A slice through a leftover axis is a slice of constant lattice index, which is a plane
        # of constant z only as long as the grid stays axis-aligned -- as a rescaling solver's
        # always is.
        plotted_dims = [spatial_dims[c] for c in cart_axes]
        leftover_spatial = [spatial_dims[d] for d in range(n_dims) if d not in cart_axes]
    else:
        plotted_dims = list(axes) if ortho else list(spatial_dims)
        leftover_spatial = (
            [coord_names[d] for d in range(n_dims) if d not in cart_axes] if ortho else []
        )

    slider_dims = [dim for dim in data.dims if dim not in plotted_dims]
    sliders = _make_sliders(data, slider_dims, leftover_spatial, template or {})

    # Held in the data's own dim order, which keeps the mesh and the field consistent with each
    # other whatever order 'cart_axes' asked for.
    mesh_dims = [d for d in data.dims if d in plotted_dims]

    # matplotlib indexes the values as (Y, X). The ortho path relabels the dims to x, y and so
    # can name that order outright; the moving path reorders both mesh and field onto the
    # lattice and so agrees with itself. A static lattice drawn on its own a1, a2 guarantees
    # neither -- the field keeps whatever order it was built in -- so the order is read back off
    # the coordinates, last axis first.
    gather: list[str] = []
    for name in reversed(axes):
        gather += [d for d in _cart_dims(data, name, factors) if d not in gather]
    flat_dims = None if (ortho or moving) else tuple(gather[:n_axes])

    mode = "moving" if moving else ("ortho" if ortho else "flat")
    order = {
        "ortho": tuple(reversed(axes)),
        "moving": tuple(mesh_dims),
        "flat": flat_dims,
    }[mode]

    def mesh_at(sel: dict) -> tuple[xr.DataArray, ...]:
        """The mesh the field is drawn on, at one position of the sliders."""
        out = []
        for name in axes:
            coord = _cart_at(data, name, factors, sel)
            out.append(coord.transpose(*mesh_dims) if moving else coord)
        return tuple(out)

    def field_at(arr: xr.DataArray, sel: dict) -> xr.DataArray:
        """One frame of a field, transposed to sit on the mesh."""
        frame_sel = {d: v for d, v in sel.items() if d in arr.dims}
        return arr.sel(frame_sel, method="nearest").transpose(*order)

    return _Drawable(
        data=data,
        mode=mode,
        axes=axes,
        plotted_dims=plotted_dims,
        sliders=sliders,
        mesh_at=mesh_at,
        field_at=field_at,
        align=align,
    )


font = {"family": "serif", "size": 12, "serif": "cmr10"}

matplotlib.rc("font", **font)
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def contour_tmpl(
    lvls: int | np.ndarray = 3,
    *,
    colors: str | np.ndarray = "gray",
    linewidths: float = 0.5,
    linestyles: str = "dashed",
) -> dict:
    """Return a contour-line template, for overlaying an outline on a filled map.

    Args:
        lvls (int or np.ndarray): If an integer, the number of contour levels; if an array, the
        height of each level.
        colors (str or np.ndarray, optional): The line colour, or one colour per level. Defaults
        to "gray".
        linewidths (float, optional): Defaults to 0.5.
        linestyles (str, optional): Defaults to "dashed".

    Returns:
        dict: The template.
    """
    return {
        "fkwargs": {
            "levels": lvls,
            "colors": colors,
            "linewidths": linewidths,
            "linestyles": linestyles,
        }
    }


def quiver_tmpl(density: int = 2, **fkwargs) -> dict:
    """Return an arrow-field template, for overlaying a vector field on a map.

    Args:
        density (int, optional): Keep every n-th arrow along each axis, to thin a crowded
        field. Defaults to 2.
        **fkwargs: Overrides passed on to Axes.quiver, e.g. color, scale, width, headwidth.

    Returns:
        dict: The template.
    """
    return {
        "fkwargs": {
            "color": "gray",
            "width": 0.009,
            "scale_units": "width",
            "scale": 0.0003,
            "pivot": "mid",
            **fkwargs,
        },
        "density": density,
    }


def _cmesh_presets() -> dict[str, dict]:
    """The colour-map presets, rebuilt on every call.

    Built fresh rather than held in a module constant because a caller is handed the dict
    itself and may well patch it, and because the norms are factories that get called once per
    map. A shared constant would leak both.

    The diverging presets use maps whose midpoint is a light neutral, so that zero reads as
    nothing. A diverging map with a coloured or dark midpoint -- 'berlin', which 'real' used to
    default to, has a midpoint of near-black -- makes zero look like an extreme value, which is
    the opposite of what a signed quantity needs.
    """
    sci = {"format": "{x:.1e}"}
    return {
        "amplitude": {
            "fkwargs": {
                "cmap": cm.oslo_r,
                "rasterized": True,
                "norm": lambda: colors.Normalize(),
            },
            "autoscale": True,
            "colorbar": {"kwargs": dict(sci)},
        },
        "amplitude - log": {
            "fkwargs": {
                "cmap": cm.oslo,
                "rasterized": True,
                "norm": lambda: colors.LogNorm(),
            },
            "autoscale": True,
            "colorbar": {"kwargs": dict(sci)},
        },
        "real": {
            "fkwargs": {
                "cmap": cm.vik,
                "rasterized": True,
                "norm": lambda: colors.CenteredNorm(),
            },
            "autoscale": True,
            "colorbar": {"kwargs": dict(sci)},
        },
        "real - log": {
            "fkwargs": {
                "cmap": cm.vik,
                "rasterized": True,
                "norm": lambda: colors.SymLogNorm(linthresh=1e-12),
            },
            "autoscale": True,
            "colorbar": {"kwargs": dict(sci)},
        },
        "phase": {
            "fkwargs": {
                "cmap": "twilight",
                "rasterized": True,
                "vmin": -np.pi,
                "vmax": np.pi,
            },
            "colorbar": {
                "kwargs": {"label": r"$\phi$"},
                "ticks": [-np.pi, 0, np.pi],
                "tickslabel": [r"$-\pi$", "0", r"$\pi$"],
            },
        },
        "potential": {
            "fkwargs": {
                "cmap": cm.cork,
                "rasterized": True,
                "norm": lambda: colors.CenteredNorm(),
            },
            "autoscale": True,
            "colorbar": {"kwargs": {"label": r"$V/E_r$", **sci}},
        },
        "difference": {
            # For a difference the sign is the whole point, so the format carries one
            "fkwargs": {
                "cmap": cm.vik,
                "rasterized": True,
                "norm": lambda: colors.CenteredNorm(),
            },
            "autoscale": True,
            "colorbar": {"kwargs": {"format": "{x:+.1e}"}},
        },
        "spin": {
            "fkwargs": {
                "cmap": cm.vik,
                "rasterized": True,
                "vmin": -1,
                "vmax": 1,
            },
            "colorbar": {
                "kwargs": {"label": r"$S_z$"},
                "ticks": [-1, 0, 1],
                "tickslabel": [r"$\sigma_-$", r"$\pi$", r"$\sigma_+$"],
            },
        },
    }


def cmesh_tmpl(
    name: str,
    *,
    cmap: "str | colors.Colormap | None" = None,
    label: str | None = None,
    clim: tuple[float, float] | None = None,
    fmt: str | None = None,
    ticks: list | None = None,
    tickslabel: list[str] | None = None,
) -> dict:
    """Return a prefilled template for a pcolormesh, contour or contourf.

    The keyword arguments cover the tweaks that otherwise get made by patching the returned
    dict, which is easy to get wrong and easy to do to a template that is shared between
    subplots. Called with a name alone, this returns exactly what it always has.

    Args:
        name (str): One of 'amplitude', 'amplitude - log', 'real', 'real - log', 'phase',
        'potential', 'difference' or 'spin'.
        cmap (str or Colormap, optional): Replaces the preset's colormap. Defaults to None.
        label (str, optional): The colorbar label. Defaults to None.
        clim (tuple[float, float], optional): Fixed colour limits. Setting these also turns
        'autoscale' off, since limits that are then rescaled away are no limits at all.
        Defaults to None.
        fmt (str, optional): The colorbar tick format, e.g. '{x:.2f}'. Defaults to None.
        ticks (list, optional): Colorbar tick positions. Defaults to None.
        tickslabel (list[str], optional): Colorbar tick labels, e.g. two poles rather than
        three for a spin map. Defaults to None.

    Returns:
        dict: The template, safe to modify further.

    Raises:
        ValueError: If 'name' is not one of the presets. It used to return None instead, which
        drew an unstyled plot and so made a mistyped name invisible.
    """
    presets = _cmesh_presets()
    if name not in presets:
        raise ValueError(
            f"unknown template {name!r}; choose one of "
            + ", ".join(repr(k) for k in presets)
        )
    temp = presets[name]

    if cmap is not None:
        temp["fkwargs"]["cmap"] = cmap
    if clim is not None:
        temp["clim"] = tuple(clim)
        temp["autoscale"] = False
    if any(v is not None for v in (label, fmt, ticks, tickslabel)):
        colorbar = temp.setdefault("colorbar", {})
        kwargs = colorbar.setdefault("kwargs", {})
        if label is not None:
            kwargs["label"] = label
        if fmt is not None:
            kwargs["format"] = fmt
        if ticks is not None:
            colorbar["ticks"] = list(ticks)
        if tickslabel is not None:
            colorbar["tickslabel"] = list(tickslabel)
    return temp


@dataclass(frozen=True)
class LayerPair:
    """Two templates for two passes over the *same* field: filled bands, then their outlines.

    Deliberately not a tuple. 'plot_eigenvector' reads a 2-tuple template as
    (field, potential-contour), so handing it a bare pair meant for one field would quietly
    style the potential's contours with the outline instead. Unpacking still works, which is
    how it is normally used:

        filled, outline = signed_log_tmpl(6)
    """

    filled: dict
    outline: dict

    def __iter__(self):
        yield self.filled
        yield self.outline


def signed_log_tmpl(
    decades: int = 6,
    *,
    label: str | None = None,
    cmap: "colors.Colormap | None" = None,
    linewidths: float = 0.5,
    outline: str = "k",
) -> LayerPair:
    """Return the pair of templates for a signed quantity spanning many decades.

    A Wannier function is the case this exists for: it is signed, it decays over six or more
    orders of magnitude, and both facts matter at once. Neither a linear scale, which shows
    only the central lobe, nor a log scale, which throws the sign away, works. The answer is
    discrete filled bands on a symmetric log ladder through zero, drawn in a diverging map with
    the band straddling zero left white so that the tails read as empty rather than as small.

    The field is expected to be normalised so that its extreme magnitude is one, as
    'w / w.max()' gives; the ladder runs from ten-to-the-minus-decades-plus-one up to one.

    Args:
        decades (int, optional): How many decades each side of zero. Defaults to 6.
        label (str, optional): The colorbar label. Defaults to None.
        cmap (Colormap, optional): The diverging map the bands are drawn from. Defaults to cm.vik.
        linewidths (float, optional): Width of the outlines. Defaults to 0.5.
        outline (str, optional): Colour of the outlines. Defaults to "k".

    Returns:
        LayerPair: '.filled' for a 'contourf' pass and '.outline' for a 'contour' pass over the
        same data, in that order. Both carry the same levels, which is what keeps the outlines
        on the band edges.
    """
    if cmap is None:
        cmap = cm.vik

    magnitudes = np.logspace(-decades + 1, 0, decades)
    levels = np.append(-magnitudes[::-1], magnitudes)

    # One colour per *band*, of which there is one fewer than there are boundaries. Sampling the
    # interior of the ramp this way reproduces the colours the hand-written version drew, which
    # passed one colour too many and relied on matplotlib discarding the last.
    band_colors = cmap(np.linspace(0, 1, 2 * decades + 1)[1:-1])
    # The band straddling zero covers everything below the smallest decade, which is noise
    band_colors[decades - 1] = [1, 1, 1, 1]

    ticklabels = [rf"$-10^{{{-e}}}$" for e in range(decades)] + [
        rf"$10^{{{e}}}$" for e in range(-decades + 1, 1)
    ]
    # ten to the zero is one, and reads better written that way
    ticklabels[0], ticklabels[-1] = "$-1$", "$1$"

    filled = {
        "fkwargs": {"levels": levels, "colors": band_colors},
        "colorbar": {"kwargs": {}, "ticks": levels, "tickslabel": ticklabels},
    }
    if label is not None:
        filled["colorbar"]["kwargs"]["label"] = label

    return LayerPair(
        filled,
        {
            "fkwargs": {
                "levels": levels,
                "colors": outline,
                "linestyles": "solid",
                "linewidths": linewidths,
            }
        },
    )


def _apply_axes(ax: Axes, spec: dict | None):
    """Apply a template's 'axes' section: the labels and aspect otherwise set by hand.

    Applied once, at setup, because none of it changes as the sliders move. Axis *limits* are
    deliberately not part of this: on a moving grid the limits are the one thing the drawing
    helpers manage themselves, frame by frame or spanned across the whole run, and a template
    setting them as well would mean two rules for one property.

    Args:
        ax (Axes): The axes to label.
        spec (dict, optional): The 'axes' entry, with any of 'xlabel', 'ylabel' and 'aspect'.
    """
    if not spec:
        return
    if spec.get("xlabel") is not None:
        ax.set_xlabel(spec["xlabel"])
    if spec.get("ylabel") is not None:
        ax.set_ylabel(spec["ylabel"])
    if spec.get("aspect") is not None:
        ax.set_aspect(spec["aspect"])


def _prepare_template(template: dict | None) -> dict:
    """A private, ready-to-draw copy of a template.

    Templates are built once and handed to several subplots, so nothing here may touch the
    caller's dict. 'create_map' used to write its 'fkwargs' default straight into whatever it
    was given, which polluted the shared mutable default of every call that passed no template
    at all.

    The norm is carried as a factory for the same reason: two maps sharing one Normalize share a
    colour scale, so autoscaling one silently rescales the other. It is called here, once per
    map, and the factory itself is left untouched for the next caller.

    Args:
        template (dict, optional): The template as the caller wrote it.

    Returns:
        dict: A deep copy, with 'fkwargs' present and its norm instantiated.
    """
    out = deepcopy(template) if template else {}
    if out.get("fkwargs") is None:
        out["fkwargs"] = {}
    norm = out["fkwargs"].get("norm")
    # A Normalize is itself callable -- it maps values into 0..1 -- so a norm passed as an
    # instance rather than as a factory would be called here with no arguments and raise a
    # TypeError from deep inside matplotlib, naming nothing that would point a caller at the
    # real problem. An instance needs no calling anyway: the deepcopy above already gave this
    # template its own, so it cannot share a colour scale with anyone else either.
    if callable(norm) and not isinstance(norm, colors.Normalize):
        out["fkwargs"]["norm"] = norm()
    return out


def _add_colorbar(fig: Figure, ax: Axes, obj, colorbar: dict | None):
    """Attach a colorbar in its own axes beside 'ax', as a template's 'colorbar' entry asks.

    The bar gets an axes of its own rather than stealing space from 'ax', so that a row of
    subplots keeps its panels the same size whether or not they carry one.

    Args:
        fig (Figure): The figure the bar belongs to.
        ax (Axes): The axes to hang it beside.
        obj: The artist whose colour scale it describes.
        colorbar (dict, optional): The template's 'colorbar' entry. None or empty draws none.

    Returns:
        Colorbar or None: The bar, or None when the template asked for none.
    """
    if not colorbar:
        return None
    cax = make_axes_locatable(ax).append_axes(
        **(colorbar.get("cax") or dict(position="right", size="5%", pad=0.05))
    )
    cbar = fig.colorbar(obj, cax=cax, **(colorbar.get("kwargs") or {"format": "{x:.1e}"}))
    if colorbar.get("tickslabel"):
        cbar.set_ticks(colorbar.get("ticks", cbar.ax.get_yticks()))
        cbar.set_ticklabels(colorbar["tickslabel"])
    return cbar


def _remove_artist(obj):
    """Take a drawn artist off its axes.

    A contour set is a bundle of collections on older matplotlib and a single artist on newer
    ones, and both forms are still met in the wild, so the removal has to cope with either.
    """
    if hasattr(obj, "collections"):
        for coll in obj.collections:
            coll.remove()
    else:
        obj.remove()


def _format_template(
    template: str | dict | tuple | NoneType, defaults: tuple[dict, ...]
) -> tuple[dict, ...]:
    """Expand whatever a caller wrote for one subplot into one template per layer.

    A subplot is drawn in layers -- the field, the potential's contours, and for
    'plot_eigenvector' the quiver on top -- and each wants its own template. Callers rarely want
    to spell all of them out, so a bare string or dict styles the first layer and the rest fall
    back to 'defaults'. The number of layers is taken from 'defaults', which is what lets
    'dashboard' (two layers) and 'plot_eigenvector' (three) share this.

    Args:
        template: A style name, a template dict, a tuple of either, or None for no styling at all.
        defaults (tuple[dict, ...]): One fallback per layer.

    Returns:
        tuple[dict, ...]: Exactly len(defaults) templates.

    Raises:
        ValueError: If 'template' is of an unusable type, or names more layers than there are.
    """
    n = len(defaults)
    if template is None:
        # Explicitly unstyled, which is not the same as omitted: no contours, no arrows either.
        return tuple({} for _ in range(n))
    if isinstance(template, str):
        given = (cmesh_tmpl(template),)
    elif isinstance(template, dict):
        given = (template,)
    elif isinstance(template, tuple):
        head = template[:1]
        given = tuple(
            t if isinstance(t, dict) else cmesh_tmpl(t) for t in head
        ) + tuple(template[1:])
    elif isinstance(template, LayerPair):
        raise ValueError(
            "signed_log_tmpl returns two passes over the same field, not a field and a "
            "potential contour, so it cannot be one template entry. Unpack it and make two "
            "calls: filled, outline = signed_log_tmpl(...)"
        )
    else:
        raise ValueError(
            "Each template entry must be a string, a dict, a tuple of those, or None; "
            f"got {type(template).__name__}"
        )
    if len(given) > n:
        raise ValueError(f"template has {len(given)} layers but only {n} are drawn")
    return (*given, *defaults[len(given):])


def get_template(name: str) -> dict:
    """Deprecated. Return a template for 'plot_eigenvector'; use 'cmesh_tmpl' instead.

    This used to return a different, older template format, with 'colormap', 'pcolormeshkwargs'
    and 'contourkwargs' keys. Nothing has read that format since the templates were reorganised
    around 'fkwargs', so a template built here silently drew unstyled. It now delegates to
    'cmesh_tmpl', which both keeps old callers working and makes them work as they read.

    Args:
        name (str): One of 'amplitude', 'amplitude - log', 'real', 'real - log' or 'phase'.

    Returns:
        dict: The template, in the current format.
    """
    warnings.warn(
        "get_template is deprecated and now returns cmesh_tmpl's format; call cmesh_tmpl "
        "directly instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return cmesh_tmpl(name)


def plot_cuts(
    eigva: xr.DataArray,
    dim: str,
    groupby: list[str] = ["band"],
    xmin: float = None,
    xmax: float = None,
    ymin: float = None,
    ymax: float = None,
    linekws: dict | list[dict] = dict(),
    figkw: dict = {},
) -> tuple[Figure, Axes]:
    """Plot a cut of a DataArray, grouping the plots by the dimensions specified in 'groupby', along the dimension 'dim'.

    Args:
        eigva (xr.DataArray): The DataArray to plot.
        dim (str): The dimension to plot against.
        groupby(list[str], optional): The dimensions to group by on the plot. Default to ['band'].
        xmin (float, optional): Like xmin from plt.plot. Defaults to None.
        xmax (float, optional): Like xmax from plt.plot. Defaults to None.
        ymin (float, optional): Like ymin from plt.plot. Defaults to None.
        ymax (float, optional): Like ymax from plt.plot. Defaults to None.
        linekws (dict or list[dict], optional): keywords arguments to be passed to plt.plot. If a list of dictionnaries are given, then the dictionary linekws[i%len(linekws)]
        is used for the i-th band. Defaults to {}.
        figkw (dict, optional): A dictionnary passed to the plt.subplots function. Defaults to {}.
    """

    list_linekw = linekws if isinstance(linekws, list) else [linekws]

    if len(groupby) > 0:
        eigva_stack = eigva.stack(stacked=groupby).squeeze()
    else:
        eigva_stack = eigva.squeeze()

    slider_dims = [d for d in eigva_stack.dims if d not in [dim, "stacked"]]
    sliders = create_sliders(eigva_stack, slider_dims)

    initial_sel = {dim: sliders[dim].value for dim in slider_dims}

    initial_band = eigva_stack.sel(initial_sel)

    fig, ax = plt.subplots(**figkw)
    lines = []

    if len(groupby) > 0:
        for i, b in enumerate(eigva_stack.stacked):
            line = ax.plot(
                eigva.coords[dim],
                initial_band.sel(stacked=b),
                **list_linekw[i % len(list_linekw)],
            )
            lines += line
    else:
        line = ax.plot(eigva.coords[dim], initial_band, **list_linekw[0])
        lines += line

    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(dim)
    ax.set_ylabel(eigva.name)

    def update(**kwargs):
        sel = {dim: kwargs[dim] for dim in sliders}
        new_bands = eigva_stack.sel(sel, method="nearest")
        pad = (new_bands.max() - new_bands.min()) * 0.05
        if len(groupby) > 0:
            for i, b in enumerate(eigva_stack.stacked):
                lines[i].set_ydata(new_bands.sel(stacked=b).data)
        else:
            lines[0].set_ydata(new_bands.data)
        if ymin is None and ymax is None:
            ax.set_ylim(new_bands.min() - pad, new_bands.max() + pad)

    out = interactive_output(update, sliders)
    # Display everything
    display(VBox(list(sliders.values()) + [out]))
    return fig, ax


def energy_levels(
    eigva: xr.DataArray,
    eigve: xr.DataArray,
    potential: Potential,
    res: int = 100,
    ymin: float = None,
    ymax: float = None,
    frac: float = 0.05,
) -> tuple[Figure, Axes]:
    """Create an interactive plot showing the mode profile of each eigenvector, placed at its corresponding energy and overlayed with the potential landscape.
    The direction taken for the eigenvector modes is controled by the sliders 'offset' and 'rotation'.

    Args:
        eigva (xr.DataArray): eigenvalues, must have a consistent shape with respect to eigve
        eigve (xr.DataArray): eigenvectors, must have a consistent shape with respect to eigva
        potential (Potential): potential object used to solve the eigenproblem.
        res (int, optional): The resolution of mode profile interpolation. Defaults to 100.
        ymin (float, optional): The lower bound of the plot, if none, it is automatically determined by looking at the potential. Defaults to None.
        ymax (float, optional): The upper bound of the plot, if none, it is automatically determined by looking at the potential. Defaults to None.
        frac (float, optional): The fraction of the plot each mode profile occupies. At larger fractions, the profiles can overlap one another. Defaults to 0.05.
    """

    band_dims = [dim for dim in eigva.dims if not dim == "band"]
    potential_dims = [
        dim for dim in potential.V.dims if dim not in ["a1", "a2", "x", "y"]
    ]

    sliders = {
        **create_sliders(eigva, band_dims),
        **create_sliders(potential.V, potential_dims),
    }

    initial_potential_sel = {dim: sliders[dim].value for dim in potential_dims}
    initial_eigve_sel = {dim: sliders[dim].value for dim in band_dims}
    initial_eigva_sel = {dim: sliders[dim].value for dim in band_dims}

    bound = float(((potential.x**2 + potential.y**2) ** 0.5).max())
    cut_coord = np.linspace(-bound, bound, res)

    slider_y = FloatSlider(
        value=0,
        min=cut_coord[0],
        max=cut_coord[-1],
        step=(cut_coord[-1] - cut_coord[0]) // res,
        description="offset",
    )

    slider_rot = FloatSlider(
        value=0,
        min=0,
        max=np.pi * 2,
        step=np.pi / 100,
        description="cut rotation (rad)",
    )

    x_coord, y_coord = (
        np.cos(slider_rot.value) * cut_coord,
        np.sin(slider_rot.value) * cut_coord + slider_y.value,
    )
    e1, e2 = (
        potential.a1 / (potential.a1 @ potential.a1),
        potential.a2 / (potential.a2 @ potential.a2),
    )

    a1_coord = xr.DataArray(x_coord * e1[0] + y_coord * e1[1], coords={"z": cut_coord})
    a2_coord = xr.DataArray(x_coord * e2[0] + y_coord * e2[1], coords={"z": cut_coord})

    initial_potential = potential.V.sel(initial_potential_sel)
    initial_eigva = eigva.sel(initial_eigva_sel)
    initial_eigve = eigve.sel(initial_eigve_sel)

    initial_potential_slice = initial_potential.interp(
        a1=a1_coord, a2=a2_coord, kwargs={"fill_value": potential.v0}
    )
    initial_eigve_slice = initial_eigve.interp(
        a1=a1_coord, a2=a2_coord, kwargs={"fill_value": 0}
    )

    potential_range = (
        initial_potential_slice.real.max() - initial_potential_slice.real.min()
    )
    ymin = initial_potential_slice.real.min() if ymin is None else ymin
    ymax = initial_potential_slice.real.max() if ymax is None else ymax
    plot_range = ymax - ymin

    initial_eigve_slice = (
        initial_eigve_slice / abs(eigve).max() * plot_range * frac + initial_eigva
    )

    pad = potential_range * 0.03
    eigve_lines = []

    fig, ax = plt.subplots()

    potential_line = ax.fill_between(
        initial_potential_slice.z,
        initial_potential_slice.real,
        initial_potential_slice.real.min() - pad,
        ec="none",
        fc="k",
        alpha=0.3,
    )

    for b in eigva.band:
        line = ax.fill_between(
            initial_eigve_slice.z,
            initial_eigve_slice.sel(band=b),
            initial_eigva.sel(band=b),
            alpha=0.5,
        )
        eigve_lines += [line]

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(cut_coord.min(), cut_coord.max())

    def update(**kwargs):
        sliders_params = {dim: kwargs[dim] for dim in kwargs if dim not in ["y", "rot"]}
        slider_y = kwargs["y"]
        slider_rot = kwargs["rot"]

        x_coord, y_coord = (
            np.cos(slider_rot) * cut_coord,
            np.sin(slider_rot) * cut_coord + slider_y,
        )

        a1_coord = xr.DataArray(
            x_coord * e1[0] + y_coord * e1[1], coords={"z": cut_coord}
        )
        a2_coord = xr.DataArray(
            x_coord * e2[0] + y_coord * e2[1], coords={"z": cut_coord}
        )

        potential_sel = {dim: sliders_params[dim] for dim in potential_dims}
        eigva_sel = {dim: sliders_params[dim] for dim in band_dims}
        eigve_sel = {dim: sliders_params[dim] for dim in band_dims}

        new_potential = potential.V.sel(potential_sel, method="nearest")
        new_eigva = eigva.sel(eigve_sel, method="nearest")
        new_eigve = eigve.sel(eigva_sel, method="nearest")

        potential_slice = new_potential.interp(
            a1=a1_coord, a2=a2_coord, kwargs={"fill_value": potential.v0}
        )
        eigve_slice = new_eigve.interp(
            a1=a1_coord, a2=a2_coord, kwargs={"fill_value": 0}
        )

        eigve_slice = eigve_slice / abs(eigve).max() * plot_range * frac + new_eigva

        potential_line.set_data(
            potential_slice.z, potential_slice.real, potential.V.real.min() - pad
        )

        for i, b in enumerate(eigva.band):
            eigve_lines[i].set_data(
                eigve_slice.z, eigve_slice.sel(band=b), new_eigva.sel(band=b)
            )

    out = interactive_output(update, {**sliders, "y": slider_y, "rot": slider_rot})
    # Display everything
    display(VBox([HBox(list(sliders.values())), HBox([slider_y, slider_rot]), out]))
    return fig, ax


def dashboard(
    eigva: xr.DataArray,
    eigvadim: str,
    eigveplots: list[list[NoneType | xr.DataArray]],
    potential: Potential,
    template: str | dict,
    titles: NoneType | list[list[NoneType | str]] = None,
    eigvawidth: int = 0.3,
    figkw: dict = {},
    gskw: dict = {},
    spines: bool = True,
    linekws: list[dict] | dict = {"color": "blue"},
    autoscale: bool = True,
    cart_axes: list[int] = [0, 1],
) -> tuple[Figure, Axes]:
    """A high-level function to plot the eigenvalues and the eigenvectors at the same time. see docs\\AtomicToMolecular.ipynb for an example.

    Args:
        eigva (xr.DataArray): The eigenvalue DataArray
        eigvadim (str): The dimension to plot the eigenvalues against
        eigveplots (list[list[Union[NoneType,xr.DataArray]]]): A matrix representing the plot structure.
        The figure will consist of a panel showing the eigenvalues, and beside it an array of pcolormeshes with the structure specified by this matrix
        potential (Potential): The potential for the contour overlay, only one needs to be given.
        template (Union[str,dict]): A template to use for the colormesh, see doc of 'plot_eigenvector' and 'get_template' for more infos.
        titles (Union[NoneType, list[list[Union[NoneType,str]]]], optional): The titles, either a matrix with the same shape as plot_matrix, or None. Defaults to None.
        eigvawidth (int, optional): The fraction of the plot taken by the eigenvalue structure. Defaults to 0.3.
        figkw (dict, optional): A dictionary to pass to the figure constructor. Defaults to {}.
        gskw (dict, optional): Additionnal keywords my be given to the gridspec handling the plots. see matplotlib doc for more infos. Defaults to {}.
        spines (bool, optional): Whether to show the box around each pcolormesh. Defaults to True.
        linekws (Union[list[dict], dict], optional): keywords arguments to be passed to plt.plot. If a list of dictionnaries are given, then the dictionary linekws[i%len(linekws)]
        is used for the i-th band. Defaults to {"color":"blue"}.
        autoscale (bool, optional): Set to False to stop the rescaling of the yaxis of the band plot. Defaults to True.
        cart_axes (list[int], optional): The two cartesian axes to plot against, with 0 = "x", 1 = "y" and 2 = "z".
        See create_map for how axes not matching the data's own native a1,a2 are handled. Defaults to [0, 1].
    """
    n_rows = len(eigveplots)
    n_cols = len(eigveplots[0])
    n_cols_tot = n_cols + eigvawidth

    list_linekw = linekws if isinstance(linekws, list) else [linekws]
    if not titles:
        titles = [[""] * n_cols for u in range(n_rows)]

    bands_dims = [dim for dim in eigva.dims if dim not in ["band", eigvadim]]
    sliders = create_sliders(eigva, bands_dims)

    fig = plt.figure(
        figsize=(min(3 * (n_cols_tot + 1), 10), max(3 * (n_rows - 1), 3)), **figkw
    )

    gs_bands = GridSpec(1, 1, left=0.05, right=eigvawidth)
    gs_eigenvectors = GridSpec(
        n_rows, n_cols, 1, left=eigvawidth + 0.05, right=0.98, **gskw
    )

    funcs: list[Callable] = []
    # Rebound rather than extended: the sliders come from the create_map calls below
    sliders = {}

    template = _format_template(template, ({}, contour_tmpl()))
    # 2D maps plot
    for i in range(n_rows):
        for j in range(n_cols):
            if eigveplots[i][j] is not None:
                plot = eigveplots[i][j]

                # A copy, not the shared template: the dashboard draws its own colorbar and
                # every subplot would otherwise switch the caller's off for good.
                ctempl = {**template[0], "colorbar": None}

                ax = fig.add_subplot(gs_eigenvectors[i, j])
                slider_ax, up, ax = create_map(
                    fig, ax, cart_axes, plot, "pcolormesh", template=ctempl
                )
                sliders.update(slider_ax)
                funcs += [up]

                slider_ax, up, ax = create_map(
                    fig, ax, cart_axes, potential.V, "contour", template=template[1]
                )
                sliders.update(slider_ax)
                funcs += [up]

                co1, co2 = coord_names[cart_axes[0]], coord_names[cart_axes[1]]
                ax.set_xlim(np.min(plot.coords[co1]), np.max(plot.coords[co1]))
                ax.set_ylim(np.min(plot.coords[co2]), np.max(plot.coords[co2]))
                ax.set_aspect("equal")
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                for sp in ["bottom", "top", "left", "right"]:
                    ax.spines[sp].set_visible(spines)
                ax.set_title(titles[i][j])

    # band plot
    ax = fig.add_subplot(gs_bands[0, 0])
    initial_eigva_sel = {dim: sliders[dim].value for dim in bands_dims}
    initial_eigva = eigva.sel(initial_eigva_sel)

    lines = []
    for i, b in enumerate(eigva.band):
        line = ax.plot(
            eigva.coords[eigvadim],
            initial_eigva.sel(band=b),
            **list_linekw[i % len(list_linekw)],
        )
        lines += line
    ax.set_xlabel(eigvadim)
    dim_pos = ax.axvline(sliders[eigvadim].value, linestyle="dashed", color="red")

    def update_eigenvectors(**kwargs):
        for f in funcs:
            f(**kwargs)

    def update_bands(**kwargs):
        sel = {dim: kwargs[dim] for dim in bands_dims}
        new_bands = eigva.sel(sel, method="nearest")
        for i, b in enumerate(eigva.band):
            lines[i].set_ydata(new_bands.sel(band=b).data)
        dim_pos.set_xdata([kwargs[eigvadim], kwargs[eigvadim]])
        if autoscale:
            pad = (new_bands.max() - new_bands.min()) * 0.05
            ax.set_ylim(new_bands.min() - pad, new_bands.max() + pad)

    def update(**kwargs):
        update_eigenvectors(**kwargs)
        update_bands(**kwargs)
        fig.canvas.draw_idle()

    out = interactive_output(update, sliders)
    # Display everything
    display(VBox(list(sliders.values()) + [out]))
    return fig, ax


def create_map(
    fig: Figure,
    ax: Axes,
    cart_axes: list[int],
    data: xr.DataArray,
    method: str,
    *,
    resolution: int | tuple[int] = None,
    template: dict = None,
    cst_bds: bool = False,
    frame: dict | None = None,
) -> tuple[dict, Callable, Axes]:
    """A low-level function to handle the creation of interactive 2D plots.

    Args:
        fig (Figure): The figure to plot the map in.
        ax (Axes): The ax to plot the map in.
        cart_axes (list[int]): The two axes to plot against, either cartesian axis indices with
        0 = "x", 1 = "y" and 2 = "z", or the names of two non-spatial dimensions to plot a field
        against its parameters instead of against space. If these aren't the data's own native
        a1,a2 axes (e.g. plotting x,z out of a 3D field, a flipped axis order, or a skewed
        lattice), the data is first re-indexed by cartesian coordinate; any spatial axis not
        selected then becomes an ordinary slider. See '_prepare' for the three layouts.
        data (xr.DataArray): The data to plot. Its cartesian coordinates may depend on the parameters
        as well as on the lattice, as a rescaling solver's do; the grid then moves with the sliders and
        is redrawn at each of their positions rather than interpolated onto a common one.
        method (str): Which matplotlib 2D plot function to use between 'pcolormesh', 'contour' and 'contourf'.
        resolution (int or tuple[int], optional): The resolution of the cartesian grid the data is
        interpolated onto, for the skewed lattices that need interpolating at all. A single int sets
        the longest axis and the others follow in proportion; a tuple sets each. The data's own
        resolution is used if None. Defaults to None.
        template (dict, optional): The template dictionnary contains all the instruction to create the plot. It has the following nested structure:
            template
                ↳ fkwargs: keyword arguments for the plotting function defined by 'method'. default to {}
                ↳ colorbar:
                    ↳ kwargs: keyword arguments passed to the Figure.colorbar function. default to {"format":"{x:.1e}"}.
                    ↳ cax: keyword arguments passed to the AxesDivider.append_axes function. Default to dict(position = 'right', size="5%", pad=0.05).
                    ↳ ticks: used to set manually the position of the colorbar ticks if necessary. Default to None.
                    ↳ tickslabel: used to set manually the text of the colorbar ticks if necessary. Default to None.
                ↳ clim: fixed colour limits, reapplied after each redraw. Default to None.
                ↳ slider_start: The initial position of the sliders. Default to 'left'.
                ↳ autoscale: Wheter to autoscale the color range. Default to True.
            The dictionnary is copied before use, so one template can be shared between subplots.
        cst_bds (bool, optional): Only meaningful on a moving grid. True keeps the axis limits fixed,
        leaving the caller to set them (plot_eigenvector spans the largest frame); False makes them
        follow the current frame, so the cloud keeps its apparent size while the axis labels change.
        Defaults to False.
        frame (dict, optional): An explicit recipe for cartesian coordinates stored as a product of
        other coordinates, e.g. {"x": ("rho_x", "lambda_x")}, overriding the array's own
        'cart_factors' attribute. Defaults to None.

    Returns:
        tuple[dict, Callable, Axes]: A slider dictionnary, an update function for interactivity and the Axes object.

    Raises:
        ValueError: If 'cart_axes' does not have exactly two entries, or 'method' is not one of
        the three supported ones.
    """
    if len(cart_axes) != 2:
        raise ValueError("create_map needs exactly 2 cart_axes")

    funcs = {
        "pcolormesh": Axes.pcolormesh,
        "contour": Axes.contour,
        "contourf": Axes.contourf,
    }
    if method not in funcs:
        raise ValueError(
            f"method must be 'pcolormesh', 'contour' or 'contourf', got {method!r}"
        )
    func = funcs[method]

    layout = _prepare(data, cart_axes, template, resolution, frame)
    sliders = layout.sliders
    template = _prepare_template(template)

    def set_limits(X: xr.DataArray, Y: xr.DataArray):
        """Fit the axes to the frame just drawn, unless the caller is spanning them itself."""
        if layout.mode == "moving" and not cst_bds:
            ax.set_xlim(float(X.min()), float(X.max()))
            ax.set_ylim(float(Y.min()), float(Y.max()))

    initial_sel = {dim: slider.value for dim, slider in sliders.items()}
    X, Y = layout.mesh_at(initial_sel)

    obj = func(ax, X, Y, layout.field_at(layout.data, initial_sel), **template["fkwargs"])
    set_limits(X, Y)
    if template.get("clim"):
        obj.set_clim(template["clim"][0], template["clim"][1])
    _add_colorbar(fig, ax, obj, template.get("colorbar"))
    _apply_axes(ax, template.get("axes"))

    # On a moving grid the mesh itself has to be rebuilt, so the artist cannot be updated in
    # place even for a pcolormesh: its geometry, not just its values, is what changed. A contour
    # set has no in-place update to speak of either. Everything else keeps its mesh -- slicing a
    # leftover axis leaves the remaining coordinates exactly as they were -- so only the values
    # need handing over.
    redraws = layout.mode == "moving" or method in ("contour", "contourf")

    def update(**kwargs):
        nonlocal obj
        sel = {dim: kwargs[dim] for dim in sliders}
        new_plot = layout.field_at(layout.data, sel)
        newX, newY = layout.mesh_at(sel) if layout.mode == "moving" else (X, Y)

        if redraws:
            _remove_artist(obj)
            obj = func(ax, newX, newY, new_plot, **template["fkwargs"])
            # The artist the colour limits were attached to no longer exists
            if template.get("clim"):
                obj.set_clim(template["clim"][0], template["clim"][1])
            set_limits(newX, newY)
        else:
            obj.set(array=new_plot.data.reshape(-1))

        if template.get("autoscale"):
            obj.autoscale()

        fig.canvas.draw_idle()

    return sliders, update, ax


def create_line(
    fig: Figure,
    ax: Axes,
    cart_axis: int,
    data: xr.DataArray,
    *,
    resolution: int | tuple[int] = None,
    template: dict = None,
    cst_bds: bool = False,
    frame: dict | None = None,
) -> tuple[dict, Callable, Axes]:
    """A low-level function to handle the creation of interactive 1D line plots, the spatial analog of
    create_map for a single cartesian axis.

    Args:
        fig (Figure): The figure to plot the line in.
        ax (Axes): The ax to plot the line in.
        cart_axis (int): The cartesian axis to plot against, with 0 = "x", 1 = "y" and 2 = "z",
        or the name of a non-spatial dimension to plot against a parameter instead.
        data (xr.DataArray): The data to plot. As in create_map, its cartesian coordinate may depend on
        the parameters, in which case the abscissa moves with the sliders.
        template (dict, optional): Only 'fkwargs' (passed to ax.plot), 'slider_start' and 'autoscale' are used.
        cst_bds (bool, optional): Only meaningful on a moving grid. False makes the x limits follow the
        current frame. Defaults to False.
        resolution (int or tuple[int], optional): The resolution to interpolate onto, for a skewed
        lattice that needs it. Defaults to None.
        frame (dict, optional): An explicit recipe for factored cartesian coordinates, as in
        create_map. Defaults to None.

    Returns:
        tuple[dict, Callable, Axes]: A slider dictionnary, an update function for interactivity and the Axes object.
    """
    layout = _prepare(data, [cart_axis], template, resolution, frame)
    sliders = layout.sliders
    template = _prepare_template(template)

    initial_sel = {dim: slider.value for dim, slider in sliders.items()}
    (X,) = layout.mesh_at(initial_sel)

    (line,) = ax.plot(
        X, layout.field_at(layout.data, initial_sel), **template["fkwargs"]
    )
    if layout.mode == "moving" and not cst_bds:
        ax.set_xlim(float(X.min()), float(X.max()))
    _apply_axes(ax, template.get("axes"))

    def update(**kwargs):
        sel = {dim: kwargs[dim] for dim in sliders}
        new_plot = layout.field_at(layout.data, sel)

        if layout.mode == "moving":
            # The abscissa moves too, so both halves of the line have to be handed over
            (newX,) = layout.mesh_at(sel)
            line.set_data(newX.data, new_plot.data)
            if not cst_bds:
                ax.set_xlim(float(newX.min()), float(newX.max()))
        else:
            line.set_ydata(new_plot.data)

        if template.get("autoscale", True):
            pad = max(1e-3, float((new_plot.max() - new_plot.min()) * 0.05))
            ax.set_ylim(float(new_plot.min() - pad), float(new_plot.max() + pad))

        fig.canvas.draw_idle()

    return sliders, update, ax


def create_quiver(
    fig: Figure,
    ax: Axes,
    cart_axes: list[int],
    dataU: xr.DataArray,
    dataV: xr.DataArray,
    *,
    resolution: int | tuple[int] = None,
    template: dict = None,
    cst_bds: bool = False,
    frame: dict | None = None,
) -> tuple[dict, Callable, Axes]:
    """A low-level function to handle the creation of interactive 2D plots using the quiver functions.

    Args:
        fig (Figure): The figure to plot the map in.
        ax (Axes): The ax to plot the map in.
        cart_axes (list[int]): The two axes to plot against, with 0 = "x", 1 = "y" and 2 = "z".
        See create_map for how axes not matching the data's own native a1,a2 are handled.
        dataU (xr.DataArray): The x-data to plot.
        dataV (xr.DataArray): The y-data to plot. It is put through whatever relabelling or
        interpolation dataU needed, so the two always end up on the same mesh.
        resolution (int or tuple[int], optional): The resolution to interpolate onto, for a skewed
        lattice that needs it. Defaults to None.
        template (dict, optional): As create_map's, plus:
            template
                ↳ density: keep every n-th arrow along each axis, to thin a crowded field. Default to 1.
        Defaults to 'quiver_tmpl()'.
        cst_bds (bool, optional): Only meaningful on a moving grid, as in create_map. Defaults to False.
        frame (dict, optional): An explicit recipe for factored cartesian coordinates. Defaults to None.

    Returns:
        tuple[dict, Callable, Axes]: A slider dictionnary, an update function for interactivity and the Axes object.

    Raises:
        ValueError: If 'cart_axes' does not have exactly two entries.
    """
    if len(cart_axes) != 2:
        raise ValueError("create_quiver needs exactly 2 cart_axes")

    if template is None:
        template = quiver_tmpl()

    layout = _prepare(dataU, cart_axes, template, resolution, frame)
    sliders = layout.sliders
    template = _prepare_template(template)
    n = template.get("density", 1)
    # The second field has to make the same journey as the first, or the arrows' two components
    # end up indexed by different axes
    dataV = layout.align(dataV)

    def subsample(arr: xr.DataArray) -> xr.DataArray:
        return arr.isel({d: slice(None, None, n) for d in arr.dims})

    def arrows(sel: dict) -> list[xr.DataArray]:
        """The positions and components of every arrow, at one position of the sliders."""
        return [
            subsample(a)
            for a in (
                *layout.mesh_at(sel),
                layout.field_at(layout.data, sel),
                layout.field_at(dataV, sel),
            )
        ]

    def set_limits(X: xr.DataArray, Y: xr.DataArray):
        if layout.mode == "moving" and not cst_bds:
            ax.set_xlim(float(X.min()), float(X.max()))
            ax.set_ylim(float(Y.min()), float(Y.max()))

    initial_sel = {dim: slider.value for dim, slider in sliders.items()}
    parts = arrows(initial_sel)
    obj = Axes.quiver(ax, *parts, **template["fkwargs"])
    set_limits(parts[0], parts[1])
    _add_colorbar(fig, ax, obj, template.get("colorbar"))
    _apply_axes(ax, template.get("axes"))

    def update(**kwargs):
        nonlocal obj
        sel = {dim: kwargs[dim] for dim in sliders}
        parts = arrows(sel)

        # A quiver carries a position per arrow as well as a value, so there is nothing useful to
        # update in place: the whole field is rebuilt either way.
        _remove_artist(obj)
        obj = Axes.quiver(ax, *parts, **template["fkwargs"])
        set_limits(parts[0], parts[1])

        if template.get("autoscale"):
            obj.autoscale()

        fig.canvas.draw_idle()

    return sliders, update, ax


def _colorscale(cmap: str | colors.Colormap, n: int = 64) -> list[list]:
    """Sample a matplotlib/cmcrameri colormap into the [[position, css color], ...] form plotly wants,
    so that the colormaps used everywhere else in this module carry over to the plotly figures."""
    cmap = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    return [[i / (n - 1), colors.to_hex(cmap(i / (n - 1)))] for i in range(n)]


def _vertex_colors(
    values: np.ndarray, cmap: colors.Colormap, crange: tuple[float, float]
) -> np.ndarray:
    """Resolve values into explicit per-vertex RGB, for the cases where plotly cannot be left to do
    the mapping itself. Returns a (n_vertices, 3) uint8 array."""
    lo, hi = crange
    normed = (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)
    normed = np.clip(np.nan_to_num(normed, nan=0.0), 0, 1)
    return (cmap(normed)[:, :3] * 255).astype(np.uint8)


def _isosurface_mesh(
    field: xr.DataArray,
    color: xr.DataArray,
    level: float,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    period: float | NoneType = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the level set of 'field' as a triangle mesh and sample 'color' on its vertices.

    Args:
        field (xr.DataArray): The field whose level set becomes the surface, dims x,y,z only.
        color (xr.DataArray): The field to sample at each vertex, dims x,y,z only. It is interpolated
        rather than read off the grid, so it needs neither the same resolution nor the same lattice as 'field'.
        level (float): The value of 'field' the surface follows.
        spacing (tuple[float, float, float]): The grid step along x, y and z.
        origin (tuple[float, float, float]): The coordinate of the grid's first cell.
        period (float, optional): Set for a cyclic color field, which 'plot_isosurface' hands over as
        the unit complex number exp(2i.pi.color/period) so that it can be interpolated without tearing.
        The angle is turned back into a value here. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The (n_vertices, 3) vertex coordinates, the
        (n_faces, 3) vertex indices, and the (n_vertices,) intensities. All three are empty if the
        level lies outside the field's range, which leaves plotly drawing nothing.
    """
    values = field.transpose("x", "y", "z").values
    # The orthogonal grid is a bounding box, so its corners fall outside the lattice and are NaN.
    # marching_cubes cannot handle those, and they would poison every cell they touch: push them
    # below the surface instead, where they simply read as 'outside'.
    values = np.nan_to_num(values, nan=float(np.nanmin(values)))

    lo, hi = float(values.min()), float(values.max())
    if not lo < level < hi:
        empty = np.zeros((0, 3))
        return empty, empty.astype(int), np.zeros(0)

    verts, faces, _, _ = marching_cubes(values, level=level, spacing=spacing)
    verts = verts + np.asarray(origin)

    # Vectorised pointwise interpolation: one intensity per vertex, wherever the vertex happens to fall
    sample = {
        axis: xr.DataArray(verts[:, i], dims="vertex")
        for i, axis in enumerate(("x", "y", "z"))
    }
    intensity = color.interp(**sample, method="linear").values
    if period is not None:
        intensity = np.angle(intensity) * period / (2 * np.pi)

    return verts, faces, intensity


def plot_isosurface(
    volume: xr.DataArray,
    color: xr.DataArray | NoneType = None,
    potential: Potential | NoneType = None,
    isovalue: float = 0.5,
    cyclic: bool | float = False,
    colorscale: str | colors.Colormap | NoneType = None,
    crange: tuple[float, float] | NoneType = None,
    resolution: int|tuple[int] = None,
    cst_bds: bool = True,
    layout: dict = {},
) -> go.FigureWidget:
    """Render a 3D mode as an interactive plotly isosurface, with the surface's shape and its color
    driven by two independent fields: 'volume' sets where the surface sits, 'color' is painted onto it.
    The archetypal use is a complex eigenvector, whose modulus gives the shape and whose phase gives
    the color, but any pair of fields sharing the potential's lattice works.

    Args:
        volume (xr.DataArray): The real field whose level set is drawn, must have exactly 3 spatial
        dims (a1, a2, a3). Pass e.g. np.abs(eigve) for a complex eigenvector. Its cartesian coordinates
        may move with the parameters, as a rescaling solver's do, in which case each frame is regridded
        onto its own bounding box as the sliders reach it.
        
        color (xr.DataArray, optional): The real field painted onto the surface, e.g. np.angle(eigve).
        It is interpolated at the surface's vertices, so it needs neither the same resolution nor the
        same lattice as 'volume'. If None, the surface is colored by 'volume' itself. Defaults to None.
        
        potential (Potential, optional): If given, one of its equipotentials is drawn as a translucent
        gray shell for context, with its own level slider. It keeps its own fixed grid even when the
        volume's moves, which is the right picture: a trap sits still in the laboratory frame while the
        cloud's frame expands around it. Defaults to None.
        
        isovalue (float, optional): The initial level, as a fraction of the selected mode's own range
        rather than an absolute value, so that a surface stays visible when switching between modes of
        very different amplitudes. Defaults to 0.5.
        
        cyclic (Union[bool, float], optional): Set this when 'color' wraps around, as a phase does.
        True means a period of 2.pi, a number gives the period explicitly. A wrapped field treated as
        an ordinary one tears along the seam where it jumps from one end of its range to the other,
        twice over: once when it is interpolated, and once when plotly blends values across the
        triangles bridging the seam. So the field is carried as a unit complex number through both
        interpolations, and the surface is then colored vertex by vertex here rather than by handing
        plotly a scale, which leaves the wrap seamless. Defaults to False.
        
        colorscale (Union[str, Colormap], optional): Any matplotlib or cmcrameri colormap, converted to
        a plotly colorscale. Defaults to the cyclic cm.romaO when 'cyclic' is set, since a wrapped field
        needs a colormap whose two ends meet, and to cm.oslo_r otherwise.
        
        crange (tuple[float, float], optional): The color limits. If None, they are taken from the whole
        'color' array, so they stay fixed as the sliders move, or from one full period when 'cyclic' is
        set. Defaults to None.
        
        resolution (int, optional): The resolution of the cartesian grid the fields are interpolated onto
        before the surface is extracted, which sets how fine the mesh is. If None, the fields' own
        resolution is used. Defaults to None.
        
        cst_bds (bool, optional): Only bites on a moving grid. True pins the scene to the largest frame
        the run reaches, so that the surface is seen to grow; False leaves plotly to rescale the box
        around each frame, which keeps the surface the same apparent size. Defaults to True.
        layout (dict, optional): Extra keyword arguments passed to the figure's update_layout. Defaults to {}.

    Returns:
        go.FigureWidget: The plotly figure widget. The sliders are displayed on their own, and the
        figure is left to the notebook to render as the cell's result, just below them, so call this
        as the last expression of a cell. Assigning the result instead keeps the sliders (which stay
        wired to the figure) but leaves it to you to display the figure where you want it.

    Raises:
        ValueError: If 'volume' isn't 3D, or if either field is complex.
    """
    spatial_dims = _spatial_dims(volume)
    if len(spatial_dims) != 3:
        raise ValueError(
            "plot_isosurface draws a surface in space and so needs a field with exactly 3 spatial "
            f"axes, got {len(spatial_dims)}. For 1D and 2D fields, use plot_eigenvector (or "
            "Potential.plot) with cart_axes instead."
        )
    for name, field in (("volume", volume), ("color", color)):
        if field is not None and np.iscomplexobj(field):
            raise ValueError(
                f"'{name}' is complex; pass a real field such as np.abs(eigve) for volume and "
                "np.angle(eigve) for color"
            )

    # np.angle drops out of xarray and returns a bare ndarray, and it is the obvious way to get a
    # phase out of a complex eigenvector, so accept that form too by re-attaching the volume's axes
    if color is not None and not isinstance(color, xr.DataArray):
        color = xr.DataArray(np.asarray(color), coords=volume.coords, dims=volume.dims)

    period = (2 * np.pi if cyclic is True else float(cyclic)) if cyclic else None

    if colorscale is None:
        colorscale = cm.romaO if period else cm.oslo_r
    # A cyclic field is colored vertex by vertex rather than through a plotly colorscale, so the
    # colormap itself is needed as well as its plotly form
    cmap = (
        matplotlib.colormaps[colorscale]
        if isinstance(colorscale, str)
        else colorscale
    )
    colorscale = _colorscale(cmap)

    # Taken before the conversion below and before the interpolation onto the cartesian grid, so the
    # limits describe the field as it was passed in. A cyclic field has to span exactly one full period,
    # otherwise the two ends of the colormap stop meeting and the wrap shows up as a seam anyway.
    if crange is None and color is not None:
        crange = (
            (-period / 2, period / 2)
            if period
            else (float(color.min()), float(color.max()))
        )

    if period is not None and color is not None:
        # A wrapped field cannot be interpolated linearly: across the seam where it jumps from
        # +period/2 back to -period/2, a linear blend runs the long way through the whole range and
        # paints a false gradient over the surface. Carrying the field as a unit complex number lets
        # it take the short way round instead, both in the regridding just below and later when the
        # vertices are sampled; _isosurface_mesh turns the angle back into a value at the very end.
        color = np.exp(2j * np.pi * color / period)

    def cart_grid(field: xr.DataArray) -> tuple[tuple, tuple]:
        """The spacing and origin marching cubes needs, read off an already-orthogonal field."""
        Xc, Yc, Zc = field.x.values, field.y.values, field.z.values
        return (
            (Xc[1] - Xc[0], Yc[1] - Yc[0], Zc[1] - Zc[0]),
            (Xc[0], Yc[0], Zc[0]),
        )

    # On a moving grid there is no single cartesian grid to interpolate onto: each frame has its own,
    # and that is the whole point, since the surface has to be seen to grow with it. So the regrid is
    # deferred into 'frame' below and redone whenever the sliders land somewhere new.
    moving = bool(_moving_dims(volume, "x", spatial_dims))

    if moving:
        color = volume if color is None else color
    else:
        volume = _to_orthogonal(volume, spatial_dims, resolution)
        color = (
            volume
            if color is None
            else _to_orthogonal(color, _spatial_dims(color), resolution)
        )
        # The surface is extracted on the volume's grid, while the color field is only ever sampled at
        # the resulting vertices, so only the former's spacing matters here
        spacing, origin = cart_grid(volume)

    if crange is None:  # the surface is colored by the volume field itself
        crange = (float(color.min()), float(color.max()))

    spatial = tuple(spatial_dims) if moving else ("x", "y", "z")
    sliders = create_sliders_from_dims(
        {d: volume.coords[d] for d in volume.dims if d not in spatial}
    )
    # The color field may be parametrised by dims the volume is not, and vice versa
    sliders.update(
        create_sliders_from_dims(
            {
                d: color.coords[d]
                for d in color.dims
                if d not in spatial and d not in sliders
            }
        )
    )
    iso_slider = FloatSlider(
        value=isovalue, min=0.01, max=0.99, step=0.01, description="isovalue"
    )

    def select(field: xr.DataArray, kwargs: dict) -> xr.DataArray:
        """Collapse every dim of 'field' that a slider drives, leaving x,y,z."""
        sel = {d: kwargs[d] for d in field.dims if d in kwargs}
        return field.sel(sel, method="nearest") if sel else field

    # Regridding a moving frame costs a full 3D interpolation, and 'update' fires on every control
    # including the isovalue, which does not move the grid at all. Holding the last few frames means
    # that sweeping the isovalue, or scrubbing back over a frame already seen, costs nothing. The
    # cache is kept small because each entry is a whole volume.
    frames: dict[tuple, tuple] = {}

    def frame(kwargs: dict) -> tuple[xr.DataArray, xr.DataArray, tuple, tuple]:
        """The volume and color fields on the current frame's own cartesian grid, with its geometry."""
        key = tuple(kwargs.get(d) for d in sliders)
        if key not in frames:
            if len(frames) >= 4:
                frames.pop(next(iter(frames)))
            vol = _to_orthogonal(select(volume, kwargs), spatial_dims, resolution)
            col = (
                vol
                if color is volume
                else _to_orthogonal(
                    select(color, kwargs), _spatial_dims(color), resolution
                )
            )
            frames[key] = (vol, col, *cart_grid(vol))
        return frames[key]

    def build(kwargs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if moving:
            field, col, sp, org = frame(kwargs)
        else:
            field, col, sp, org = select(volume, kwargs), select(color, kwargs), spacing, origin
        lo, hi = float(field.min()), float(field.max())
        return _isosurface_mesh(
            field,
            col,
            lo + kwargs["isovalue"] * (hi - lo),
            sp,
            org,
            period,
        )

    def mesh_data(
        verts: np.ndarray, faces: np.ndarray, intensity: np.ndarray
    ) -> dict:
        """The per-frame part of the mode's trace, shared by the initial draw and the updates."""
        data = dict(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
        )
        if period is None:
            data["intensity"] = intensity
        else:
            # A cyclic field cannot be handed over as a scalar intensity either: plotly blends
            # intensities linearly across each triangle, so the ones bridging the seam would still
            # sweep backwards through the whole colormap and draw it as a line on the surface.
            # Resolving the colors here avoids that, because the two ends of a cyclic colormap are
            # the same color and blending across the seam simply stays on it.
            data["vertexcolor"] = _vertex_colors(intensity, cmap, crange)
        return data

    initial = {d: s.value for d, s in {**sliders, "isovalue": iso_slider}.items()}
    style = dict(
        flatshading=False,
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.2, roughness=0.5),
        name="mode",
    )
    if period is None:
        style.update(
            colorscale=colorscale,
            cmin=crange[0],
            cmax=crange[1],
            colorbar=dict(title=color.name or "", thickness=15),
        )

    fig = go.FigureWidget(data=[go.Mesh3d(**mesh_data(*build(initial)), **style)])

    pot_slider = None
    if potential is not None:
        pot = _to_orthogonal(potential.V, _spatial_dims(potential.V), resolution).real
        pot_spacing = (
            pot.x.values[1] - pot.x.values[0],
            pot.y.values[1] - pot.y.values[0],
            pot.z.values[1] - pot.z.values[0],
        )
        pot_origin = (pot.x.values[0], pot.y.values[0], pot.z.values[0])
        pot_slider = FloatSlider(
            value=0.3, min=0.01, max=0.99, step=0.01, description="V level"
        )
        sliders.update(
            create_sliders_from_dims(
                {
                    d: pot.coords[d]
                    # The shell is interpolated up front whatever the volume does, so its own spatial
                    # dims are x,y,z here even when 'spatial' is the volume's moving lattice
                    for d in pot.dims
                    if d not in ("x", "y", "z") and d not in sliders
                }
            )
        )

        def build_potential(kwargs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            field = select(pot, kwargs)
            lo, hi = float(field.min()), float(field.max())
            return _isosurface_mesh(
                field,
                field,
                lo + kwargs["Vlevel"] * (hi - lo),
                pot_spacing,
                pot_origin,
            )

        pverts, pfaces, _ = build_potential({**initial, "Vlevel": pot_slider.value})
        fig.add_trace(
            go.Mesh3d(
                x=pverts[:, 0],
                y=pverts[:, 1],
                z=pverts[:, 2],
                i=pfaces[:, 0],
                j=pfaces[:, 1],
                k=pfaces[:, 2],
                color="gray",
                opacity=0.15,
                hoverinfo="skip",
                flatshading=False,
                showscale=False,
                name="potential",
            )
        )

    if period is not None:
        # Coloring the surface by hand leaves plotly with no scale to build a colorbar from, so it
        # gets one of its own: an empty trace that draws nothing and carries only the color scale.
        # Added last, so that the mode and the potential keep trace indices 0 and 1.
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(
                    colorscale=colorscale,
                    cmin=crange[0],
                    cmax=crange[1],
                    color=[crange[0]],
                    showscale=True,
                    colorbar=dict(title=color.name or "", thickness=15),
                ),
                hoverinfo="skip",
                showlegend=False,
                name="colorbar",
            )
        )

    scene = dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z",
        # Without this the box is stretched to a cube and the lattice's proportions are lost
        aspectmode="data",
    )
    if moving and cst_bds:
        # Left to itself plotly refits the box around each frame, so a surface that doubles in size
        # looks unchanged. The coordinates span every frame the run went through, so their extremes
        # are the largest one: pinning the scene there is what makes the growth visible.
        for name in ("x", "y", "z"):
            coord = volume.coords[name]
            scene[f"{name}axis_range"] = [float(coord.min()), float(coord.max())]

    fig.update_layout(
        scene=scene,
        margin=dict(l=0, r=0, t=0, b=0),
        **layout,
    )

    def update(**kwargs):
        with fig.batch_update():
            fig.data[0].update(**mesh_data(*build(kwargs)))
            if pot_slider is not None:
                pverts, pfaces, _ = build_potential(kwargs)
                fig.data[1].update(
                    x=pverts[:, 0],
                    y=pverts[:, 1],
                    z=pverts[:, 2],
                    i=pfaces[:, 0],
                    j=pfaces[:, 1],
                    k=pfaces[:, 2],
                )

    controls = {**sliders, "isovalue": iso_slider}
    if pot_slider is not None:
        controls["Vlevel"] = pot_slider

    out = interactive_output(update, controls)
    # Only the controls are displayed here, the figure is the return value and so is rendered by the
    # notebook as the cell's result, just below them. Putting it in this VBox as well would show it
    # twice, since a FigureWidget handed back from a cell is displayed again.
    display(VBox(list(controls.values()) + [out]))
    return fig


def plot_eigenvector(
    plots: list[list[xr.DataArray | NoneType]],
    potentials: list[list[Potential | NoneType]],
    templates: list[list[str | tuple[str | dict | NoneType]]] | NoneType = None,
    quivers: NoneType | list[list[NoneType | tuple[xr.DataArray]]] = None,
    ncontours: int = 3,
    cart_axes: list[int] | list[list[list[int]]] = [0, 1],
    resolutions: list[list[int|tuple[int]|NoneType]] = None,
    cst_bds: bool = False,
) -> tuple[Figure, list[Axes]]:
    """The main function to plot eigenvectors in a interactive manner.

    Args:
        plots (list[list[xr.DataArray]]): A list of list of eigenvectors xr.DataArrays, each DataArray will be plotted in a separate subplot,
        in a grid-pattern determined by the structure of the list of lists.
        potentials (list[list[Union[Potential, NoneType]]]): The potentials to be plotted as contour for each plot.
        templates (list[list[Union[str,tuple[Union[str, dict]]]]]): A dictionnary containing all the instruction to define each subplot style. the templates can also be strings
        calling predefined templates, such as 'amplitude', 'phase', 'real' and more. see 'get_template' for more informations.
        titles (Union[NoneType, list[list[str]]): The title for each subplot. Default to None.
        quivers (Union[NoneType, list[list[Union[NoneType, tuple[xr.DataArray]]]]], optional): An optional argument to overlay quiver plots on top of the eigenvectors.
        Each entry of the list of lists must either be None or contain a tuple of DataArrays (U,V,C), see the quiver function from matplotlib for more informations.
        Only valid for subplots whose cart_axes has 2 entries. Defaults to None.
        ncontours (int, optional): The number of contours to use for the plot. This is a convenience argument. For more control, see the template formalism. Default to 3
        cart_axes (Union[list[int], list[list[list[int]]]], optional): The cartesian axes to plot against, with 0 = "x", 1 = "y" and 2 = "z".
        Either a single list of axes (e.g. [0, 1]) applied to every subplot, or a matrix with the same shape as 'plots' giving each
        subplot its own axes list independently. A single axis gives line plots (potential and eigenvector overlaid, styled with
        simple defaults rather than the template system); two axes give the usual pcolormesh/contour/quiver overlay. Any spatial
        axis not selected becomes an extra slider. Defaults to [0, 1].
        cst_bds (bool, optional): Only useful on a field whose cartesian coordinates move with the parameters,
        as a rescaling solver's do. True holds the axes at the largest frame the run ever reaches, so that
        the cloud is seen to grow; False lets them follow the current frame, so the cloud keeps its apparent
        size and it is the axis labels that change. Defaults to False.

    Raises:
        ValueError: Raise errors if the shapes are not consistent, or if quivers are given for a subplot with a single cart_axis.
    """
    n_rows = len(plots)
    n_cols = len(plots[0])

    if templates is None:
        templates = [[None] * n_cols for _ in range(n_rows)]
    if resolutions is None:
        resolutions = [[None] * n_cols for _ in range(n_rows)]

    if len(templates) != n_rows or len(potentials) != n_rows:
        raise ValueError("different shapes for plots and templates")
    if len(templates[0]) != n_cols or len(potentials[0]) != n_cols:
        raise ValueError("different shapes for plots and templates")
    for i in range(1, n_rows):
        if len(plots[i]) != n_cols:
            raise ValueError(f"Length of row {i} of 'plots' not consistent")
        if len(templates[i]) != n_cols:
            raise ValueError(f"Length of row {i} of 'templates' not consistent")
        if len(potentials[i]) != n_cols:
            raise ValueError(f"Length of row {i} of 'potentials' not consistent")

    if quivers is None:
        quivers = [[None] * n_cols for u in range(n_rows)]

    funcs: list[Callable] = []
    sliders = {}

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        squeeze=False,
        figsize=(3 * (n_cols + 1), 3 * n_rows),
        layout="tight",
    )

    if isinstance(cart_axes[0], int):
        cart_axes = [[cart_axes] * n_cols for _ in range(n_rows)]
        
    
    elif len(cart_axes) != n_rows or any(len(row) != n_cols for row in cart_axes):
        raise ValueError(
            "cart_axes, given as a matrix, must have the same shape as plots"
        )

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i][j]
            template = _format_template(
                templates[i][j], ({}, contour_tmpl(ncontours), quiver_tmpl())
            )

            plot = plots[i][j]
            poten = potentials[i][j]
            quiv = quivers[i][j]
            cart_axe = cart_axes[i][j]
            resolution = resolutions[i][j]

            if len(cart_axe) == 1 and quiv is not None:
                raise ValueError(
                    f"quivers require 2 cart_axes; quiver plots have no 1D analog (cell [{i}][{j}])"
                )

            if len(cart_axe) == 1:
                if plot is not None:
                    slids, up, ax = create_line(
                        fig,
                        ax,
                        cart_axe[0],
                        plot,
                        resolution=resolution,
                        template=template[0],
                        cst_bds=cst_bds,
                    )
                    sliders.update(slids)
                    funcs += [up]
                if poten is not None:
                    # Potential and eigenvector are on very different scales: give the potential its own y-axis
                    ax_pot = ax.twinx()
                    slids, up, ax_pot = create_line(
                        fig,
                        ax_pot,
                        cart_axe[0],
                        poten.V,
                        resolution=resolution,
                        template={
                            "fkwargs": {
                                "color": "gray",
                                "linestyle": "dashed",
                                "linewidth": 1,
                            }
                        },
                        cst_bds=cst_bds,
                    )
                    ax_pot.set_ylabel("Potential", color="gray")
                    ax_pot.tick_params(axis="y", colors="gray")
                    sliders.update(slids)
                    funcs += [up]
            else:
                if plot is not None:
                    slids, up, ax = create_map(
                        fig,
                        ax,
                        cart_axe,
                        plot,
                        "pcolormesh",
                        resolution=resolution,
                        template=template[0],
                        cst_bds=cst_bds,
                    )
                    sliders.update(slids)
                    funcs += [up]
                if poten is not None:
                    slids, up, ax = create_map(
                        fig,
                        ax,
                        cart_axe,
                        poten.V,
                        "contour",
                        resolution=resolution,
                        template=template[1],
                        cst_bds=cst_bds,
                    )
                    sliders.update(slids)
                    funcs += [up]
                if quiv is not None:
                    slids, up, ax = create_quiver(
                        fig,
                        ax,
                        cart_axe,
                        quiv[0],
                        quiv[1],
                        resolution=resolution,
                        template=template[2],
                        cst_bds=cst_bds,
                    )
                    sliders.update(slids)
                    funcs += [up]

            bounds = (
                plot if plot is not None else (poten.V if poten is not None else None)
            )
            # On a moving grid the axes have to be pinned to the largest frame the run reaches,
            # which is what makes the cloud visibly grow. When cst_bds is not set, the create_*
            # update functions refit the limits frame by frame instead, and a grid that does not
            # move has one box either way, so cst_bds does not enter into it.
            if bounds is not None and cst_bds:
                axis_names = _axis_names(cart_axe)
                bound_factors = _cart_factors(bounds)
                moves = bool(
                    _moving_dims(
                        bounds, axis_names[0], _spatial_dims(bounds), bound_factors
                    )
                )
                if moves:
                    ax.set_xlim(*_cart_extent(bounds, axis_names[0], bound_factors))
                    if len(cart_axe) == 2:
                        ax.set_ylim(*_cart_extent(bounds, axis_names[1], bound_factors))
            # Equal by default, since a field on a lattice should be drawn in real
            # proportions, but not when a template has asked for something else: a map over two
            # parameters has no reason to be square.
            if len(cart_axe) == 2 and not any(
                (layer.get("axes") or {}).get("aspect") for layer in template
            ):
                ax.set_aspect("equal")
            axes[i][j] = ax

    def update(**kwargs):
        for f in funcs:
            f(**kwargs)

    out = interactive_output(update, sliders)
    # Display everything
    display(VBox(list(sliders.values()) + [out]))
    return fig, axes


if __name__ == "__main__":
    # Imported here rather than at module level: only the demo below needs them, and keeping
    # plotting free of a solver dependency avoids a cycle if a solver ever imports plotting.
    from bloch_schrodinger.fdsolver import FDSolver
    from bloch_schrodinger.potential import create_parameter

    foo = Potential([[5, 0], [0, 5]], (50, 50))

    omega = create_parameter("omega", np.linspace(3, 10, 5))

    foo.set((foo.x**2 + 1.0000001 * foo.y**2) * omega)

    bar = FDSolver(foo, 1 / 2)

    eigva, eigve = bar.solve(5)

    conttmpl = contour_tmpl()

    plot_eigenvector(
        [[abs(eigve) ** 2, eigve.real]],
        [[foo, foo]],
        [[("amplitude", conttmpl), ({"fkwargs":{"color":"k"}})]],
        cart_axes=[[[0, 1], [0]]],
    )
    plt.show()

