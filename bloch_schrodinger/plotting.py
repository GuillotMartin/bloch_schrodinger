import numpy as np
import xarray as xr
from types import NoneType
from typing import Union, Callable
from ipywidgets import FloatSlider, HBox, VBox, interactive_output
from IPython.display import display
import cmcrameri.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy

from bloch_schrodinger.potential import Potential
from bloch_schrodinger.utils import create_sliders, create_sliders_from_dims

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import QuadMesh
from matplotlib.quiver import Quiver
from matplotlib.contour import QuadContourSet
from matplotlib.gridspec import GridSpec


font = {"family": "serif", "size": 12, "serif": "cmr10"}

matplotlib.rc("font", **font)
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def contour_tmpl()->dict:
    """Return a simple contour plot template for general purpose use.

    Returns:
        dict
    """
    return {"fkwargs":{
        "colors": "gray",
        "linewidths": 0.5,
        "linestyles": "dashed",
    }}

def quiver_tmpl()->dict:
    """Return a simple contour plot template for general purpose use.

    Returns:
        dict
    """
    return {'fkwargs':{
                "color": "gray",
                "width": 0.009,
                "scale_units": "width",
                "scale": 0.0003,
                "pivot": "mid",
            },
            "density": 2
    }
    
def cmesh_tmpl(name:str)->dict:
    """Create simple prefiled templates for a pcolormesh object.

    Args:
        name (str): a template chosen between "amplitude", "amplitude - log", "real", "real - log" and "phase"

    Returns:
        dict: _description_
    """
      
    
    if name == "amplitude":
        temp = {
            "fkwargs":{
                "cmap": cm.oslo_r,
                "rasterized":True,
                "norm": lambda: (
                    colors.Normalize()
                ),  # using a factory function to avoid colormap sharing
            },
            "autoscale":True,
            "colorbar":{"kwargs":{"format": "{x:.1e}"}}
        }
        return temp

    if name == "amplitude - log":
        temp = {
            "fkwargs":{
                "cmap": cm.oslo,
                "rasterized":True,
                "norm": lambda: (
                    colors.LogNorm()
                ),  # using a factory function to avoid colormap sharing
            },
            "autoscale":True,
            "colorbar":{"kwargs":{"format": "{x:.1e}"}}
        }
        return temp

    if name == "real":
        temp = {
            "fkwargs":{
                "cmap": cm.berlin,
                "rasterized":True,
                "norm": lambda: (
                    colors.CenteredNorm()
                ),  # using a factory function to avoid colormap sharing
            },
            "autoscale": True,
            "colorbar":{"kwargs":{"format": "{x:.1e}"}}
        }
        return temp

    if name == "real - log":
        temp = {
            "fkwargs":{
                "cmap": cm.berlin,
                "rasterized":True,
                "norm": lambda: (
                    colors.SymLogNorm()
                ),  # using a factory function to avoid colormap sharing
            },
            "autoscale": True,
            "colorbar":{"kwargs":{"format": "{x:.1e}"}}
        }
        return temp

    if name == "phase":
        temp = {
            "fkwargs":{
                "cmap": 'twilight',
                "rasterized":True,
                "vmin": -np.pi,
                "vmax": np.pi
            },
            "colorbar":{
                "kwargs":{
                    "label": r"$\phi$", 
                },
                "ticks": [-np.pi, 0, np.pi],
                "tickslabel":[r"$-\pi$", "0", r"$\pi$"],
            }
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
        "density": 2
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
    linekws: Union[dict, list[dict]] = dict(),
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
    eigveplots: list[list[Union[NoneType, xr.DataArray]]],
    potential: Potential,
    template: Union[str, dict],
    titles: Union[NoneType, list[list[Union[NoneType, str]]]] = None,
    eigvawidth: int = 0.3,
    figkw: dict = {},
    gskw: dict = {},
    spines: bool = True,
    linekws: Union[list[dict], dict] = {"color": "blue"},
    autoscale: bool = True,
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

    axes = []
    funcs:list[Callable] = []
    sliders = {}


    def make_tmpl(template:Union[str, dict])->dict:
        """Check wheter template is a string or a dict, and if a str, create the proper dictionnary.
        """
        return template if isinstance(template, dict) else cmesh_tmpl(template)

    def format_template(template:tuple[Union[str, dict]])->tuple[dict, dict]:
        """Format a template input into the proper tuple"""
        if isinstance(template, str):
            template = (cmesh_tmpl(template), contour_tmpl())
        elif not isinstance(template, tuple):
            raise ValueError("Each template entry must either be a tuple or a string")
        elif len(template) == 1:
            ctmpl =  make_tmpl(template[0])
            template = (ctmpl, contour_tmpl())
        elif len(template) == 2:
            ctmpl =  make_tmpl(template[0])
            template = (ctmpl, template[1])
        return template

    template = format_template(template)
    # 2D maps plot
    for i in range(n_rows):
        for j in range(n_cols):
            if eigveplots[i][j] is not None:

                plot = eigveplots[i][j]
                
                ctempl = template[0]
                ctempl['colorbar'] = None
                
                ax = fig.add_subplot(gs_eigenvectors[i, j])
                slider_ax, up, ax = create_map(fig, ax, 'x', 'y', plot, 'pcolormesh', ctempl)
                sliders.update(slider_ax)
                funcs += [up]

                slider_ax, up, ax = create_map(fig, ax, 'x', 'y', potential.V, 'contour', template[1])
                sliders.update(slider_ax)
                funcs += [up]
                
                ax.set_xlim(np.min(plot.x), np.max(plot.x))
                ax.set_ylim(np.min(plot.y), np.max(plot.y))
                ax.set_aspect("equal")
                
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
    dim1: str,
    dim2: str,
    data: xr.DataArray,
    method: str,
    template: dict = {},
) -> tuple[dict, Callable, Axes]:
    """A low-level function to handle the creation of interactive 2D plots

    Args:
        fig (Figure): The figure to plot the map in.
        ax (Axes): The ax to plot the map in
        dim1 (str): coordinate along the x-axis of the plot
        dim2 (str): coordinate along the y-axis of the plot
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
    
    if method == 'pcolormesh':
        func = Axes.pcolormesh
    elif method == 'contour':
        func = Axes.contour
    elif method == 'contourf':
        func = Axes.contourf
            
    # Creating the sliders objects
    slider_dims = [dim for dim in data.dims if dim not in ["a1", "a2", dim1, dim2]]
    sliders = create_sliders_from_dims({dim:data.coords[dim] for dim in slider_dims}, start = template.get('slider_start', 'left'))
    
    # Creating the fkwargs key just in case, to avoid testing its existence every time
    if template.get('fkwargs') is None:
        template['fkwargs'] = {}
        
    # Initial parameter selections
    initial_field_sel = {dim: sliders[dim].value for dim in sliders}
    
    #Extracting the norm object from the template, it is convoluted to avoid unintended sharing of colorscales
    template = deepcopy(template) # We are going to mutate template so copying is important
    if template.get('fkwargs'):
        if template['fkwargs'].get('norm'):
            template['fkwargs']["norm"] = template['fkwargs']["norm"]() if callable(template['fkwargs']["norm"]) else template['fkwargs']["norm"]
    
    # Shortcut names for the coordinates
    X = data.coords[dim1]
    Y = data.coords[dim2]
    
    # Initial data selection
    plot_init = data.sel(initial_field_sel)

    obj = func(ax,
        X, Y, plot_init, **template['fkwargs']
    )
    
    colorbar = template.get('colorbar')
    if colorbar is not None:
        if colorbar.get('kwargs') is None:
            colorbar['kwargs'] = {'format':"{x:.1e}"}
        
    if colorbar:
        divider = make_axes_locatable(ax)
        if colorbar.get('cax') is None:
            colorbar['cax'] = dict(position = 'right', size="5%", pad=0.05)
        cax = divider.append_axes(**colorbar['cax'])
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

        field_sel = {
            dim: value
            for dim, value in sel.items()
        }

        new_plot = data.sel(field_sel, method="nearest")

        if method in ['contour', 'contourf']:
            if hasattr(obj, "collections"):
                for coll in obj.collections:
                    coll.remove()
            else:
                obj.remove()
                
            obj = func(ax,
                X, Y, new_plot, **template['fkwargs']
            )
        else:
            obj.set(array = new_plot.data.reshape(-1))

        if template.get("autoscale"):
            obj.autoscale()

        fig.canvas.draw_idle()

    return sliders, update, ax

def create_quiver(
    fig: Figure,
    ax: Axes,
    dim1: str,
    dim2: str,
    dataU: xr.DataArray,
    dataV: xr.DataArray,
    template: dict = quiver_tmpl(),
) -> tuple[dict, Callable, Axes]:
    """A low-level function to handle the creation of interactive 2D plots using the quiver functions

    Args:
        fig (Figure): The figure to plot the map in.
        ax (Axes): The ax to plot the map in
        dim1 (str): coordinate along the x-axis of the plot
        dim2 (str): coordinate along the y-axis of the plot
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
    template = deepcopy(template)
    func = Axes.quiver
    n = template.get('density',1)
    # Creating the sliders objects
    slider_dims = [dim for dim in dataU.dims if dim not in ["a1", "a2", dim1, dim2]]
    sliders = create_sliders_from_dims({dim:dataU.coords[dim] for dim in slider_dims}, start = template.get('slider_start', 'left'))
    
    # Creating the fkwargs key just in case, to avoid testing its existence every time
    if template.get('fkwargs') is None:
        template['fkwargs'] = {}
        
    # Initial parameter selections
    initial_field_sel = {dim: sliders[dim].value for dim in sliders}
    
    #Extracting the norm object from the template, it is convoluted to avoid unintended sharing of colorscales
    if template.get('fkwargs'):
        if template['fkwargs'].get('norm'):
            template['fkwargs']["norm"] = template['fkwargs']["norm"]() if callable(template['fkwargs']["norm"]) else template['fkwargs']["norm"]
    
    # Shortcut names for the coordinates
    X = dataU.coords[dim1]
    Y = dataU.coords[dim2]
    
    # Initial data selection
    plot_init_U = dataU.sel(initial_field_sel)
    plot_init_V = dataV.sel(initial_field_sel)

    obj = func(ax,
        X[::n, ::n], Y[::n, ::n], 
        plot_init_U[::n, ::n], plot_init_V[::n, ::n], 
        **template['fkwargs']
    )
    
    colorbar = template.get('colorbar')
    if colorbar is not None:
        if colorbar.get('kwargs') is None:
            colorbar['kwargs'] = {'format':"{x:.1e}"}
        
    if colorbar:
        divider = make_axes_locatable(ax)
        if colorbar.get('cax') is None:
            colorbar['cax'] = dict(position = 'right', size="5%", pad=0.05)
        cax = divider.append_axes(**colorbar['cax'])
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

        field_sel = {
            dim: value
            for dim, value in sel.items()
        }

        new_plot_U = dataU.sel(field_sel, method="nearest")
        new_plot_V = dataV.sel(field_sel, method="nearest")

        if hasattr(obj, "collections"):
            for coll in obj.collections:
                coll.remove()
        else:
            obj.remove()
            
        obj = func(ax,
            X[::n, ::n], Y[::n, ::n], 
            new_plot_U[::n, ::n], new_plot_V[::n, ::n], 
            **template['fkwargs']
        )

        if template.get("autoscale"):
            obj.autoscale()

        fig.canvas.draw_idle()

    return sliders, update, ax

def plot_eigenvector(
    plots: list[list[Union[xr.DataArray, NoneType]]],
    potentials: list[list[Union[Potential, NoneType]]],
    templates: list[list[Union[str,tuple[Union[str, dict]]]]],
    quivers: Union[NoneType, list[list[Union[NoneType, tuple[xr.DataArray]]]]] = None,
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
        Defaults to None.

    Raises:
        ValueError: Raise errors if the shapes are not consistent.
    """
    n_rows = len(plots)
    n_cols = len(plots[0])
    
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

    funcs:list[Callable] = []
    sliders = {}
    
    def make_tmpl(template:Union[str, dict])->dict:
        """Check wheter template is a string or a dict, and if a str, create the proper dictionnary.
        """
        return template if isinstance(template, dict) else cmesh_tmpl(template)
    
    def format_template(template:tuple[Union[str, dict]])->tuple[dict, dict, dict]:
        """Format a template input into the proper tuple"""
        if isinstance(template, str):
            template = (cmesh_tmpl(template), contour_tmpl(), quiver_tmpl())
        elif isinstance(template, dict):
            template = (template, contour_tmpl(), quiver_tmpl())
        elif not isinstance(template, tuple):
            raise ValueError("Each template entry must either be a tuple or a string")
        elif len(template) == 1:
            ctmpl =  make_tmpl(template[0])
            template = (ctmpl, contour_tmpl(), quiver_tmpl())
        elif len(template) == 2:
            ctmpl =  make_tmpl(template[0])
            template = (ctmpl, template[1], quiver_tmpl())
        elif len(template) == 3:
            ctmpl =  make_tmpl(template[0])
            template = (ctmpl, template[1], template[2])
        return template
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        squeeze=False,
        figsize=(3 * (n_cols + 1), 3 * n_rows),
        layout="tight",
    )
    

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i][j]
            template = format_template(templates[i][j])
            
            plot = plots[i][j]
            poten = potentials[i][j]
            quiv = quivers[i][j]
            if plot is not None:
                slids, up, ax = create_map(fig, ax, "x", "y", plot, "pcolormesh", template[0])
            sliders.update(slids)
            funcs += [up]
            if poten is not None:
                slids, up, ax = create_map(fig, ax, "x", "y", poten.V, "contour", template[1])
            sliders.update(slids)
            funcs += [up]
            if quiv is not None:
                slids, up, ax = create_quiver(fig, ax, "x", "y", quiv[0], quiv[1], template[2])
            sliders.update(slids)
            funcs += [up]
            
            ax.set_xlim(np.min(plot.x), np.max(plot.x))
            ax.set_ylim(np.min(plot.y), np.max(plot.y))
            ax.set_aspect("equal")           
            axes[i][j] = ax
            
                        

    def update(**kwargs):
        for f in funcs:
            f(**kwargs)

    out = interactive_output(update, sliders)
    # Display everything
    display(VBox(list(sliders.values()) + [out]))
    return fig, axes



from bloch_schrodinger.potential import create_parameter  # noqa: E402
from bloch_schrodinger.fdsolver import FDSolver  # noqa: E402
if __name__ == '__main__':
    
    foo = Potential(
        [[5,0], [0,5]], (50,50)
    )
    
    omega = create_parameter('omega', np.linspace(3,10,5))
    
    foo.set(
        (foo.x**2 + foo.y**2)*omega
    )
    
    
    bar = FDSolver(
        foo, 1/2
    )
    
    eigva, eigve = bar.solve(5)
    
    conttmpl = contour_tmpl()
    
    plot_eigenvector(
        [[abs(eigve)**2, eigve.real]],
        [[foo, foo]],
        [[('amplitude', conttmpl), 'real']]
    )
    plt.show()
    
