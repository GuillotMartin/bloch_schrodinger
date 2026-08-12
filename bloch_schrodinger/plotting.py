from collections.abc import Callable
from copy import deepcopy
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
    data: xr.DataArray, spatial_dims: list[str], resolution: int = None
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


font = {"family": "serif", "size": 12, "serif": "cmr10"}

matplotlib.rc("font", **font)
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def contour_tmpl(lvls: int | np.ndarray = 3) -> dict:
    """Return a simple contour plot template for general purpose use.

    Args:
        lvls (Union[int, np.ndarray]): If an integer, the number of contour levels, if an array, each entry specifies the height of a level.
    Returns:
        dict
    """
    return {
        "fkwargs": {
            "levels": lvls,
            "colors": "gray",
            "linewidths": 0.5,
            "linestyles": "dashed",
        }
    }


def quiver_tmpl() -> dict:
    """Return a simple contour plot template for general purpose use.

    Returns:
        dict
    """
    return {
        "fkwargs": {
            "color": "gray",
            "width": 0.009,
            "scale_units": "width",
            "scale": 0.0003,
            "pivot": "mid",
        },
        "density": 2,
    }


def cmesh_tmpl(name: str) -> dict:
    """Create simple prefiled templates for a pcolormesh object.

    Args:
        name (str): a template chosen between "amplitude", "amplitude - log", "real", "real - log" and "phase"

    Returns:
        dict: _description_
    """

    if name == "amplitude":
        temp = {
            "fkwargs": {
                "cmap": cm.oslo_r,
                "rasterized": True,
                "norm": lambda: (
                    colors.Normalize()
                ),  # using a factory function to avoid colormap sharing
            },
            "autoscale": True,
            "colorbar": {"kwargs": {"format": "{x:.1e}"}},
        }
        return temp

    if name == "amplitude - log":
        temp = {
            "fkwargs": {
                "cmap": cm.oslo,
                "rasterized": True,
                "norm": lambda: (
                    colors.LogNorm()
                ),  # using a factory function to avoid colormap sharing
            },
            "autoscale": True,
            "colorbar": {"kwargs": {"format": "{x:.1e}"}},
        }
        return temp

    if name == "real":
        temp = {
            "fkwargs": {
                "cmap": cm.berlin,
                "rasterized": True,
                "norm": lambda: (
                    colors.CenteredNorm()
                ),  # using a factory function to avoid colormap sharing
            },
            "autoscale": True,
            "colorbar": {"kwargs": {"format": "{x:.1e}"}},
        }
        return temp

    if name == "real - log":
        temp = {
            "fkwargs": {
                "cmap": cm.berlin,
                "rasterized": True,
                "norm": lambda: colors.SymLogNorm(
                    linthresh=1e-12
                ),  # using a factory function to avoid colormap sharing
            },
            "autoscale": True,
            "colorbar": {"kwargs": {"format": "{x:.1e}"}},
        }
        return temp

    if name == "phase":
        temp = {
            "fkwargs": {
                "cmap": "twilight",
                "rasterized": True,
                "vmin": -np.pi,
                "vmax": np.pi,
            },
            "colorbar": {
                "kwargs": {
                    "label": r"$\phi$",
                },
                "ticks": [-np.pi, 0, np.pi],
                "tickslabel": [r"$-\pi$", "0", r"$\pi$"],
            },
        }
        return temp


def get_template(name: str) -> dict:
    """Return a pre-filled template made to be used with the 'plot_eigenvector' function. Includes the argument 'contourkwargs', 'pcolormeshkwargs',
    'cbarkwargs' and 'quiverkwargs' that are passed to the according matplotlib functions.

    Args:
        name (str): The name of the template, right now, 'amplitude', 'real', 'amplitude - log', 'real - log' and 'phase are implemented.
    """

    quivers = {
        "color": "gray",
        "width": 0.009,
        "scale_units": "width",
        "scale": 0.0003,
        "pivot": "mid",
        "density": 2,
    }

    contours = {
        "levels": None,
        "colors": "white",
        "linewidths": 0.3,
        "linestyles": "dashed",
    }

    if name == "amplitude":
        temp = {
            "colormap": cm.oslo_r,
            "norm": lambda: (
                colors.Normalize()
            ),  # using a factory function to avoid colormap sharing
            "colorbarticks": None,
            "autoscale": True,
            "pcolormeshkwargs": {},
            "contourkwargs": deepcopy(contours),
            "cbarkwargs": {
                "format": "{x:.1e}",
            },
            "quiverkwargs": deepcopy(quivers),
        }
        temp["contourkwargs"]["colors"] = "black"
        return temp

    if name == "amplitude - log":
        temp = {
            "colormap": cm.oslo,
            "norm": lambda: colors.LogNorm(),
            "colorbarticks": None,
            "autoscale": True,
            "pcolormeshkwargs": {},
            "contourkwargs": deepcopy(contours),
            "cbarkwargs": {
                "format": "{x:.1e}",
            },
            "quiverkwargs": deepcopy(quivers),
        }
        return temp

    if name == "real":
        temp = {
            "colormap": cm.berlin,
            "norm": lambda: colors.CenteredNorm(),
            "colorbarticks": None,
            "autoscale": True,
            "pcolormeshkwargs": {},
            "contourkwargs": deepcopy(contours),
            "cbarkwargs": {
                "format": "{x:.1e}",
            },
            "quiverkwargs": deepcopy(quivers),
        }
        return temp

    if name == "real - log":
        temp = {
            "colormap": cm.berlin,
            "norm": lambda: colors.SymLogNorm(1e-12),
            "colorbarticks": None,
            "autoscale": True,
            "pcolormeshkwargs": {},
            "contourkwargs": deepcopy(contours),
            "cbarkwargs": {
                "format": "{x:.1e}",
            },
            "quiverkwargs": deepcopy(quivers),
        }
        return temp

    if name == "phase":
        temp = {
            "colormap": "twilight",
            "norm": lambda: colors.CenteredNorm(),
            "clim": (-np.pi, np.pi),
            "pcolormeshkwargs": {},
            "contourkwargs": deepcopy(contours),
            "cbarkwargs": {"label": r"$\phi$", "ticks": [-np.pi, 0, np.pi]},
            "cbartickslabel": [r"$-\pi$", "0", r"$\pi$"],
            "quiverkwargs": deepcopy(quivers),
        }
        return temp


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

    def make_tmpl(template: str | dict) -> dict:
        """Check wheter template is a string or a dict, and if a str, create the proper dictionnary."""
        return template if isinstance(template, dict) else cmesh_tmpl(template)

    def format_template(template: tuple[str | dict]) -> tuple[dict, dict]:
        """Format a template input into the proper tuple"""
        if isinstance(template, str):
            template = (cmesh_tmpl(template), contour_tmpl())
        elif not isinstance(template, tuple):
            raise ValueError("Each template entry must either be a tuple or a string")
        elif len(template) == 1:
            ctmpl = make_tmpl(template[0])
            template = (ctmpl, contour_tmpl())
        elif len(template) == 2:
            ctmpl = make_tmpl(template[0])
            template = (ctmpl, template[1])
        return template

    template = format_template(template)
    # 2D maps plot
    for i in range(n_rows):
        for j in range(n_cols):
            if eigveplots[i][j] is not None:
                plot = eigveplots[i][j]

                ctempl = template[0]
                ctempl["colorbar"] = None

                ax = fig.add_subplot(gs_eigenvectors[i, j])
                slider_ax, up, ax = create_map(
                    fig, ax, cart_axes, plot, "pcolormesh", ctempl
                )
                sliders.update(slider_ax)
                funcs += [up]

                slider_ax, up, ax = create_map(
                    fig, ax, cart_axes, potential.V, "contour", template[1]
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
    template: dict = {},
) -> tuple[dict, Callable, Axes]:
    """A low-level function to handle the creation of interactive 2D plots

    Args:
        fig (Figure): The figure to plot the map in.
        ax (Axes): The ax to plot the map in
        cart_axes (list[int]): The two cartesian axes to plot against, with 0 = "x", 1 = "y" and 2 = "z".
        If these aren't the data's own native a1,a2 axes (e.g. plotting x,z out of a 3D field, a flipped
        axis order, or a skewed lattice), the data is first interpolated onto an orthogonal cartesian grid;
        any spatial axis not selected then becomes an ordinary slider.
        data (xr.DataArray): The data to plot.
        method (str): Which matplotlib 2D plot function to use between 'pcolormesh', 'contour' and 'contourf'.
        template (dict, optional): The template dictionnary contains all the instruction to create the plot. It has the following nested structure:
            template
                ↳ fkwargs: keyword arguments for the plotting function defined by 'method'. default to {}
                ↳ colorbar:
                    ↳ kwargs: keyword arguments passed to the Figure.colorbar function. default to {"format":"{x:.1e}"}.
                    ↳ cax: keyword arguments passed to the AxesDivider.append_axes function. Default to dict(position = 'right', size="5%", pad=0.05).
                    ↳ ticks: used to set manually the position of the colorbar ticks if necessary. Default to None.
                    ↳ tickslabel: used to set manually the text of the colorbar ticks if necessary. Default to None.
                ↳ slider_start: The initial position of the sliders. Default to 'left'.
                ↳ autoscale: Wheter to autoscale the color range. Default to True.

    Returns:
        tuple[dict, Callable, Axes]: A slider dictionnary, an update function for interactivity and the Axes object.
    """
    if len(cart_axes) != 2:
        raise ValueError("create_map needs exactly 2 cart_axes")

    if method == "pcolormesh":
        func = Axes.pcolormesh
    elif method == "contour":
        func = Axes.contour
    elif method == "contourf":
        func = Axes.contourf
    else:
        raise ValueError(
            f"method must be 'pcolormesh', 'contour' or 'contourf', got {method!r}"
        )

    spatial_dims = _spatial_dims(data)
    n_dims = len(spatial_dims)
    ortho = cart_axes != [0, 1] or n_dims != 2
    if ortho:
        data = _to_orthogonal(data, spatial_dims)

    dim1, dim2 = coord_names[cart_axes[0]], coord_names[cart_axes[1]]
    plotted_dims = [dim1, dim2] if ortho else spatial_dims
    leftover_spatial = (
        [coord_names[d] for d in range(n_dims) if d not in cart_axes] if ortho else []
    )

    # Creating the sliders objects
    slider_dims = [dim for dim in data.dims if dim not in plotted_dims]
    sliders = _make_sliders(data, slider_dims, leftover_spatial, template)

    # Creating the fkwargs key just in case, to avoid testing its existence every time
    if template.get("fkwargs") is None:
        template["fkwargs"] = {}

    # Initial parameter selections
    initial_field_sel = {dim: sliders[dim].value for dim in sliders}

    # Extracting the norm object from the template, it is convoluted to avoid unintended sharing of colorscales
    template = deepcopy(
        template
    )  # We are going to mutate template so copying is important
    if template.get("fkwargs"):
        if template["fkwargs"].get("norm"):
            template["fkwargs"]["norm"] = (
                template["fkwargs"]["norm"]()
                if callable(template["fkwargs"]["norm"])
                else template["fkwargs"]["norm"]
            )

    # Shortcut names for the coordinates
    X = data.coords[dim1]
    Y = data.coords[dim2]

    # Initial data selection
    plot_init = data.sel(initial_field_sel, method="nearest")
    if ortho:
        plot_init = plot_init.transpose(dim2, dim1)

    obj = func(ax, X, Y, plot_init, **template["fkwargs"])

    if template.get("clim"):
        obj.set_clim(template["clim"][0], template["clim"][1])

    colorbar = template.get("colorbar")
    if colorbar is not None:
        if colorbar.get("kwargs") is None:
            colorbar["kwargs"] = {"format": "{x:.1e}"}

    if colorbar:
        divider = make_axes_locatable(ax)
        if colorbar.get("cax") is None:
            colorbar["cax"] = dict(position="right", size="5%", pad=0.05)
        cax = divider.append_axes(**colorbar["cax"])
        cbar = fig.colorbar(
            obj,
            cax=cax,
            **colorbar["kwargs"],
        )
        if colorbar.get("tickslabel"):
            ticks = colorbar.get("ticks", cbar.ax.get_yticks())
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(colorbar["tickslabel"])

    def update(**kwargs):

        nonlocal obj
        sel = {dim: kwargs[dim] for dim in sliders}

        field_sel = {dim: value for dim, value in sel.items()}

        new_plot = data.sel(field_sel, method="nearest")
        if ortho:
            new_plot = new_plot.transpose(dim2, dim1)

        if method in ["contour", "contourf"]:
            if hasattr(obj, "collections"):
                for coll in obj.collections:
                    coll.remove()
            else:
                obj.remove()

            obj = func(ax, X, Y, new_plot, **template["fkwargs"])
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
    template: dict = {},
) -> tuple[dict, Callable, Axes]:
    """A low-level function to handle the creation of interactive 1D line plots, the spatial analog of
    create_map for a single cartesian axis.

    Args:
        fig (Figure): The figure to plot the line in.
        ax (Axes): The ax to plot the line in.
        cart_axis (int): The cartesian axis to plot against, with 0 = "x", 1 = "y" and 2 = "z".
        data (xr.DataArray): The data to plot.
        template (dict, optional): Only 'fkwargs' (passed to ax.plot), 'slider_start' and 'autoscale' are used.

    Returns:
        tuple[dict, Callable, Axes]: A slider dictionnary, an update function for interactivity and the Axes object.
    """
    spatial_dims = _spatial_dims(data)
    n_dims = len(spatial_dims)
    ortho = n_dims != 1
    if ortho:
        data = _to_orthogonal(data, spatial_dims)

    dim1 = coord_names[cart_axis]
    plotted_dims = [dim1] if ortho else spatial_dims
    leftover_spatial = (
        [coord_names[d] for d in range(n_dims) if d != cart_axis] if ortho else []
    )

    slider_dims = [dim for dim in data.dims if dim not in plotted_dims]
    sliders = _make_sliders(data, slider_dims, leftover_spatial, template)

    template = deepcopy(template)
    if template.get("fkwargs") is None:
        template["fkwargs"] = {}

    initial_field_sel = {dim: sliders[dim].value for dim in sliders}

    X = data.coords[dim1]
    plot_init = data.sel(initial_field_sel, method="nearest")

    (line,) = ax.plot(X, plot_init, **template["fkwargs"])

    def update(**kwargs):
        sel = {dim: kwargs[dim] for dim in sliders}
        new_plot = data.sel(sel, method="nearest")
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
    template: dict = quiver_tmpl(),
) -> tuple[dict, Callable, Axes]:
    """A low-level function to handle the creation of interactive 2D plots using the quiver functions

    Args:
        fig (Figure): The figure to plot the map in.
        ax (Axes): The ax to plot the map in
        cart_axes (list[int]): The two cartesian axes to plot against, with 0 = "x", 1 = "y" and 2 = "z".
        See create_map for how axes not matching the data's own native a1,a2 are handled.
        dataU (xr.DataArray): The x-data to plot.
        dataV (xr.DataArray): The y-data to plot.
        template (dict, optional): The template dictionnary contains all the instruction to create the plot. It has the following nested structure:
            template
                ↳ fkwargs: keyword arguments for the plotting function defined by 'method'. default to {}
                ↳ colorbar:
                    ↳ kwargs: keyword arguments passed to the Figure.colorbar function. default to {"format":"{x:.1e}"}.
                    ↳ cax: keyword arguments passed to the AxesDivider.append_axes function. Default to dict(position = 'right', size="5%", pad=0.05).
                    ↳ ticks: used to set manually the position of the colorbar ticks if necessary. Default to None.
                    ↳ tickslabel: used to set manually the text of the colorbar ticks if necessary. Default to None.
                ↳ slider_start: The initial position of the sliders. Default to 'left'.
                ↳ autoscale: Wheter to autoscale the color range. Default to True.

    Returns:
        tuple[dict, Callable, Axes]: A slider dictionnary, an update function for interactivity and the Axes object.
    """
    if len(cart_axes) != 2:
        raise ValueError("create_quiver needs exactly 2 cart_axes")

    template = deepcopy(template)
    func = Axes.quiver
    n = template.get("density", 1)

    spatial_dims = _spatial_dims(dataU)
    n_dims = len(spatial_dims)
    ortho = cart_axes != [0, 1] or n_dims != 2
    if ortho:
        dataU = _to_orthogonal(dataU, spatial_dims)
        dataV = _to_orthogonal(dataV, spatial_dims)

    dim1, dim2 = coord_names[cart_axes[0]], coord_names[cart_axes[1]]
    plotted_dims = [dim1, dim2] if ortho else spatial_dims
    leftover_spatial = (
        [coord_names[d] for d in range(n_dims) if d not in cart_axes] if ortho else []
    )

    # Creating the sliders objects
    slider_dims = [dim for dim in dataU.dims if dim not in plotted_dims]
    sliders = _make_sliders(dataU, slider_dims, leftover_spatial, template)

    # Creating the fkwargs key just in case, to avoid testing its existence every time
    if template.get("fkwargs") is None:
        template["fkwargs"] = {}

    # Initial parameter selections
    initial_field_sel = {dim: sliders[dim].value for dim in sliders}

    # Extracting the norm object from the template, it is convoluted to avoid unintended sharing of colorscales
    if template.get("fkwargs"):
        if template["fkwargs"].get("norm"):
            template["fkwargs"]["norm"] = (
                template["fkwargs"]["norm"]()
                if callable(template["fkwargs"]["norm"])
                else template["fkwargs"]["norm"]
            )

    # Shortcut names for the coordinates
    X = dataU.coords[dim1]
    Y = dataU.coords[dim2]

    def subsample(arr):
        return arr.isel({d: slice(None, None, n) for d in arr.dims})

    # Initial data selection
    plot_init_U = dataU.sel(initial_field_sel, method="nearest")
    plot_init_V = dataV.sel(initial_field_sel, method="nearest")
    if ortho:
        plot_init_U = plot_init_U.transpose(dim2, dim1)
        plot_init_V = plot_init_V.transpose(dim2, dim1)

    obj = func(
        ax,
        subsample(X),
        subsample(Y),
        subsample(plot_init_U),
        subsample(plot_init_V),
        **template["fkwargs"],
    )

    colorbar = template.get("colorbar")
    if colorbar is not None:
        if colorbar.get("kwargs") is None:
            colorbar["kwargs"] = {"format": "{x:.1e}"}

    if colorbar:
        divider = make_axes_locatable(ax)
        if colorbar.get("cax") is None:
            colorbar["cax"] = dict(position="right", size="5%", pad=0.05)
        cax = divider.append_axes(**colorbar["cax"])
        cbar = fig.colorbar(
            obj,
            cax=cax,
            **colorbar["kwargs"],
        )
        if colorbar.get("tickslabel"):
            ticks = colorbar.get("ticks", cbar.ax.get_yticks())
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(colorbar["tickslabel"])

    def update(**kwargs):

        nonlocal obj
        sel = {dim: kwargs[dim] for dim in sliders}

        field_sel = {dim: value for dim, value in sel.items()}

        new_plot_U = dataU.sel(field_sel, method="nearest")
        new_plot_V = dataV.sel(field_sel, method="nearest")
        if ortho:
            new_plot_U = new_plot_U.transpose(dim2, dim1)
            new_plot_V = new_plot_V.transpose(dim2, dim1)

        if hasattr(obj, "collections"):
            for coll in obj.collections:
                coll.remove()
        else:
            obj.remove()

        obj = func(
            ax,
            subsample(X),
            subsample(Y),
            subsample(new_plot_U),
            subsample(new_plot_V),
            **template["fkwargs"],
        )

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
    resolution: int = None,
    layout: dict = {},
) -> go.FigureWidget:
    """Render a 3D mode as an interactive plotly isosurface, with the surface's shape and its color
    driven by two independent fields: 'volume' sets where the surface sits, 'color' is painted onto it.
    The archetypal use is a complex eigenvector, whose modulus gives the shape and whose phase gives
    the color, but any pair of fields sharing the potential's lattice works.

    Args:
        volume (xr.DataArray): The real field whose level set is drawn, must have exactly 3 spatial
        dims (a1, a2, a3). Pass e.g. np.abs(eigve) for a complex eigenvector.
        color (xr.DataArray, optional): The real field painted onto the surface, e.g. np.angle(eigve).
        It is interpolated at the surface's vertices, so it needs neither the same resolution nor the
        same lattice as 'volume'. If None, the surface is colored by 'volume' itself. Defaults to None.
        potential (Potential, optional): If given, one of its equipotentials is drawn as a translucent
        gray shell for context, with its own level slider. Defaults to None.
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

    volume = _to_orthogonal(volume, spatial_dims, resolution)
    color = (
        volume
        if color is None
        else _to_orthogonal(color, _spatial_dims(color), resolution)
    )
    if crange is None:  # the surface is colored by the volume field itself
        crange = (float(color.min()), float(color.max()))

    # The surface is extracted on the volume's grid, while the color field is only ever sampled at
    # the resulting vertices, so only the former's spacing matters here
    Xc, Yc, Zc = volume.x.values, volume.y.values, volume.z.values
    spacing = (Xc[1] - Xc[0], Yc[1] - Yc[0], Zc[1] - Zc[0])
    origin = (Xc[0], Yc[0], Zc[0])

    spatial = ("x", "y", "z")
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

    def build(kwargs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        field = select(volume, kwargs)
        lo, hi = float(field.min()), float(field.max())
        return _isosurface_mesh(
            field,
            select(color, kwargs),
            lo + kwargs["isovalue"] * (hi - lo),
            spacing,
            origin,
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
                    for d in pot.dims
                    if d not in spatial and d not in sliders
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

    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            # Without this the box is stretched to a cube and the lattice's proportions are lost
            aspectmode="data",
        ),
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
    templates: [list[list[str | tuple[str | dict | NoneType]]] | NoneType] = None,
    quivers: NoneType | list[list[NoneType | tuple[xr.DataArray]]] = None,
    ncontours: int = 3,
    cart_axes: list[int] | list[list[list[int]]] = [0, 1],
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

    Raises:
        ValueError: Raise errors if the shapes are not consistent, or if quivers are given for a subplot with a single cart_axis.
    """
    n_rows = len(plots)
    n_cols = len(plots[0])

    if templates is None:
        templates = [[None] * n_cols for _ in range(n_rows)]

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

    def make_tmpl(template: str | dict) -> dict:
        """Check wheter template is a string or a dict, and if a str, create the proper dictionnary."""
        return template if isinstance(template, dict) else cmesh_tmpl(template)

    def format_template(template: tuple[str | dict | NoneType]) -> tuple[dict, dict, dict]:
        """Format a template input into the proper tuple"""
        if template is None:
            template = ({}, {}, {})
        elif isinstance(template, str):
            template = (cmesh_tmpl(template), contour_tmpl(ncontours), quiver_tmpl())
        elif isinstance(template, dict):
            template = (template, contour_tmpl(ncontours), quiver_tmpl())
        elif not isinstance(template, tuple):
            raise ValueError("Each template entry must either be a tuple or a string")
        elif len(template) == 1:
            ctmpl = make_tmpl(template[0])
            template = (ctmpl, contour_tmpl(ncontours), quiver_tmpl())
        elif len(template) == 2:
            ctmpl = make_tmpl(template[0])
            template = (ctmpl, template[1], quiver_tmpl())
        elif len(template) == 3:
            ctmpl = make_tmpl(template[0])
            template = (ctmpl, template[1], template[2])
        return template

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
            template = format_template(templates[i][j])

            plot = plots[i][j]
            poten = potentials[i][j]
            quiv = quivers[i][j]
            cart_axe = cart_axes[i][j]

            if len(cart_axe) == 1 and quiv is not None:
                raise ValueError(
                    f"quivers require 2 cart_axes; quiver plots have no 1D analog (cell [{i}][{j}])"
                )

            if len(cart_axe) == 1:
                if plot is not None:
                    slids, up, ax = create_line(fig, ax, cart_axe[0], plot)
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
                        {
                            "fkwargs": {
                                "color": "gray",
                                "linestyle": "dashed",
                                "linewidth": 1,
                            }
                        },
                    )
                    ax_pot.set_ylabel("Potential", color="gray")
                    ax_pot.tick_params(axis="y", colors="gray")
                    sliders.update(slids)
                    funcs += [up]
            else:
                if plot is not None:
                    slids, up, ax = create_map(
                        fig, ax, cart_axe, plot, "pcolormesh", template[0]
                    )
                    sliders.update(slids)
                    funcs += [up]
                if poten is not None:
                    slids, up, ax = create_map(
                        fig, ax, cart_axe, poten.V, "contour", template[1]
                    )
                    sliders.update(slids)
                    funcs += [up]
                if quiv is not None:
                    slids, up, ax = create_quiver(
                        fig, ax, cart_axe, quiv[0], quiv[1], template[2]
                    )
                    sliders.update(slids)
                    funcs += [up]

            bounds = (
                plot if plot is not None else (poten.V if poten is not None else None)
            )
            if bounds is not None:
                co1 = coord_names[cart_axe[0]]
                ax.set_xlim(np.min(bounds.coords[co1]), np.max(bounds.coords[co1]))
                if len(cart_axe) == 2:
                    co2 = coord_names[cart_axe[1]]
                    ax.set_ylim(np.min(bounds.coords[co2]), np.max(bounds.coords[co2]))
            if len(cart_axe) == 2:
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
        [[("amplitude", conttmpl), "real"]],
        cart_axes=[[[0, 1], [0]]],
    )
    plt.show()
