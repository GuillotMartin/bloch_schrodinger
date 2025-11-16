import numpy as np
import xarray as xr
from types import NoneType
from typing import Union, Type
from ipywidgets import interact, FloatSlider, IntSlider, HBox, VBox, Layout, interactive_output
from IPython.display import display
import cmcrameri.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import QuadMesh
from matplotlib.quiver import Quiver
from matplotlib.contour import QuadContourSet



font = {"family": "serif", "size": 12, "serif": "cmr10"}

matplotlib.rc("font", **font)
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def get_template(name:str):
    """Return a pre-filled template made to be used with the 'plot_eigenvector' function. Includes the argument 'contourkwargs', 'pcolormeshkwargs', 
    'cbarkwargs' and 'quiverkwargs' that are passed to the according matplotlib functions.

    Args:
        name (str): The name of the template, right now, 'amplitude', 'real', 'amplitude - log' and 'phase are implemented.
    """
    
    quivers = {
        "color":"white",
        "width":0.009,
        "scale_units":'width',
        "scale":0.0003,
        "pivot":'mid'
    }
    
    if name == 'amplitude':
        temp = {
            "colormap": cm.oslo_r,
            "norm": lambda: colors.Normalize(), # using a factory function to avoid colormap sharing
            "colorbarticks":None,
            "pcolormeshkwargs":{},
            "contourkwargs":{
                "levels": 0, 
                "colors": 'black', 
                "linewidths": 1, 
                "linestyles": 'dashed',
            },
            "cbarkwargs":{
                "format": "{x:.1e}",
            },
            "quiverkwargs":deepcopy(quivers),
        }
        return temp

    if name == 'amplitude - log':
        temp = {
            "colormap": cm.oslo,
            "norm": lambda: colors.LogNorm(),
            "colorbarticks":None,
            "pcolormeshkwargs":{},
            "contourkwargs":{
                "levels": 0, 
                "colors": 'white', 
                "linewidths": 1, 
                "linestyles": 'dashed',
            },
            "cbarkwargs":{
                "format": "{x:.1e}",
            },
            "quiverkwargs":deepcopy(quivers),
        }
        return temp
    
    if name== 'real':
        temp = {
            "colormap": cm.berlin,
            "norm": lambda: colors.CenteredNorm(),
            "colorbarticks":None,
            "pcolormeshkwargs":{},
            "contourkwargs":{
                "levels": 0, 
                "colors": 'white', 
                "linewidths": 1, 
                "linestyles": 'dashed',
            },
            "cbarkwargs":{
                "format": "{x:.1e}",
            },
            "quiverkwargs":deepcopy(quivers),
        }
        return temp
    
    if name== 'phase':
        temp = {
            "colormap": 'twilight',
            "norm": lambda: colors.CenteredNorm(), 
            "clim": (-np.pi, np.pi),
            "pcolormeshkwargs":{},
            "contourkwargs":{
                "levels": 0, 
                "colors": 'white', 
                "linewidths": 1, 
                "linestyles": 'dashed',
            },
            "cbarkwargs":{
                "label": r'$\phi$', 
                "ticks": [-np.pi, 0, np.pi]
            },
            "cbartickslabel":[r"$-\pi$", "0", r"$\pi$"],
            "quiverkwargs":deepcopy(quivers),
        }
        return temp

def plot_eigenvector(
    plots:list[list[xr.DataArray]], 
    potentials:list[list[xr.DataArray]], 
    templates:list[list[Union[str,dict]]],
    titles: list[list[str]],
    quivers:Union[NoneType, list[list[Union[NoneType, tuple[xr.DataArray]]]]] = None
    ):
    """The main function to plot eigenvectors in a interactive manner.

    Args:
        plots (list[list[xr.DataArray]]): A list of list of eigenvectors xr.DataArrays, each DataArray will be plotted in a separate subplot, 
        in a grid-pattern determined by the structure of the list of lists.
        potentials (list[list[xr.DataArray]]): The potentials to be plotted as contour for each plot.
        templates (list[list[Union[str,dict]]]): A dictionnary containing all the instruction to define each subplot style. the templates can also be strings 
        calling predefined templates, such as 'amplitude', 'phase', 'real' and more. see 'get_template' for more informations.
        titles (list[list[str]]): The title for each subplot.
        quivers (Union[NoneType, list[list[Union[NoneType, tuple[xr.DataArray]]]]], optional): An optional argument to overlay quiver plots on top of the eigenvectors. 
        Each entry of the list of lists must either be None or contain a tuple of DataArrays (U,V,C), see the quiver function from matplotlib for more informations. 
        Defaults to None.

    Raises:
        ValueError: Raise errors if the shapes are not consistent.
    """
    n_rows = len(plots)
    n_cols = len(plots[0])
    if len(templates) != n_rows or len(potentials) != n_rows: raise ValueError("different shapes for plots and templates")
    if len(templates[0]) != n_cols or len(potentials[0]) != n_cols: raise ValueError("different shapes for plots and templates")
    for i in range(1, n_rows):
        if len(plots[i]) != n_cols: raise ValueError(f"Length of row {i} of 'plots' not consistent")
        if len(templates[i]) != n_cols: raise ValueError(f"Length of row {i} of 'templates' not consistent")
        if len(potentials[i]) != n_cols: raise ValueError(f"Length of row {i} of 'potentials' not consistent")
    
    if not quivers:
        quivers = [[None]*n_cols for u in range(n_rows)]
        
    sliders = {}
      
    def create_axe(
            fig:Figure, 
            ax:Axes, 
            plot:xr.DataArray, 
            potential:xr.DataArray, 
            quiver:Union[NoneType, tuple[xr.DataArray]], 
            template:Union[str,dict], 
            title:str
        )-> tuple[Axes, dict, QuadMesh, QuadContourSet, Quiver, dict]:
        """A function to create a single subplot

        Args:
            fig (Figure): the global figure object
            ax (Axes): The ax in which to plot.
            plot (xr.DataArray): The DataArray to plot as a heatmap.
            potential (xr.DataArray): The Potential to plot as a contour plot.
            quiver (Union[NoneType, tuple[xr.DataArray]]): The DataArray tuple to plot as a quiver.
            template (Union[str,dict]): The template, either a string to call 'get_template' or a dictionary
            title (str): The title of the subplot.

        Returns:
            tuple[Axes, dict, QuadMesh, QuadContourSet, Quiver, dict]: The axes, objects plotted and a template in the dictionary format
        """
        slider_dims = [dim for dim in plot.dims if dim not in ['a1','a2','x','y']]
        sliders_ax = {}
        for dim in slider_dims:
            coord = plot.coords[dim].values
            val = coord[0]  # start left
            if np.issubdtype(coord.dtype, np.floating):
                sliders_ax[dim] = FloatSlider(
                    min=float(coord.min()),
                    max=float(coord.max()),
                    step=float((coord.max() - coord.min()) / max(100, len(coord))),
                    value=float(val),
                    description=dim
                )
            else:
                sliders_ax[dim] = IntSlider(min=0, max=len(coord) - 1, step=1, value=coord[0], description=dim)

        initial_field_sel = {dim:sliders_ax[dim].value for dim in slider_dims}
        initial_potential_sel = {dim:sliders_ax[dim].value for dim in slider_dims if dim in potential.V.dims}
        
        
        field = plot.sel(initial_field_sel)
        pot = potential.V.sel(initial_potential_sel)

        if isinstance(template,str):
            template = get_template(template)

        norm = template['norm']() if callable(template['norm']) else template['norm']
        mesh = ax.pcolormesh(
            plot.x, plot.y, field, 
            shading='auto', 
            cmap = template['colormap'],
            norm = norm,
            **template["pcolormeshkwargs"],
        )
                
        contour = ax.contour(
            plot.x, plot.y, pot, 
            **template["contourkwargs"]
        )

        quiv = None
        if quiver is not None:
            initial_quiver_sel = {dim:sliders_ax[dim].value for dim in slider_dims if dim in quiver[0].dims}
            svert, shor = quiver[0].sel(initial_quiver_sel).shape
            narrows = template.get("number of arrows", (max(1,svert//15)))
            template["number of arrows"] = narrows
            UVC = [C.sel(initial_quiver_sel)[::narrows, ::narrows] for C in quiver]
            
            quiv = ax.quiver(
                plot.x[::narrows, ::narrows], plot.y[::narrows, ::narrows], *UVC,
                **template.get("quiverkwargs", dict(color='w'))
            )
            
        
        ax.set_aspect('equal')
        ax.set_title(title)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(
            mesh, 
            cax=cax,
            **template['cbarkwargs'],
        )
        if template.get("cbartickslabel"):
            ticks = template["cbarkwargs"].get("ticks", cbar.ax.get_yticks())
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(template["cbartickslabel"])
            
        return ax, template, mesh, contour, quiv, sliders_ax
        # cbar.set_label("Potential")
    
    fig, axes = plt.subplots(
        nrows = n_rows, ncols = n_cols, squeeze=False,
        figsize=(3*(n_cols + 1), 3*n_rows),
        layout = 'tight',
    )
    # fig.subplots_adjust(wspace=3)
    indx, jndx, meshes, contours, quivs, dict_templates = [], [], [], [], [], []
    
    for i in range(n_rows):
        for j in range(n_cols):
            ax, template, mesh, contour, quiv, slider_ax = create_axe(
                fig,
                axes[i][j], 
                plots[i][j], 
                potentials[i][j],
                quivers[i][j],
                templates[i][j],
                titles[i][j]
            )
            
            dict_templates += [deepcopy(template)]
            axes[i][j] = ax
            indx += [i]
            jndx += [j]
            meshes += [mesh]
            contours += [contour]
            quivs += [quiv]
            sliders.update(slider_ax)

    def update(**kwargs):
        sel = {dim:kwargs[dim] for dim in sliders}
        
        new_plots, new_potentials, new_quivs = [], [], []
        for i in range(n_rows):
            for j in range(n_cols):
                field_sel = {dim:value for dim, value in sel.items() 
                             if dim in plots[i][j].dims 
                             and dim not in ['a1','a2','x','y']}
                
                potential_sel = {dim:value for dim, value in field_sel.items() 
                                 if dim in potentials[i][j].V.dims}
                
                new_plots += [plots[i][j].sel(field_sel, method = 'nearest')]
                new_potentials += [potentials[i][j].V.sel(potential_sel, method = 'nearest')]
                
                if quivers[i][j] is not None:
                    new_quiv = [C.sel(field_sel, method = 'nearest') for C in quivers[i][j]]
                else:
                    new_quiv = None
                new_quivs += [new_quiv]
                

        for u, (new_plot, new_pot, new_quiv) in enumerate(zip(new_plots, new_potentials, new_quivs)):
            
            meshes[u].set_array(new_plot.data.reshape(-1))
            template = dict_templates[u]
            if 'clim' not in template:
                meshes[u].autoscale()
                     
            contours[u].remove()
            contours[u] = axes[indx[u]][jndx[u]].contour(
                new_plot.x, new_plot.y, new_pot, 
                **template['contourkwargs']
            )
            if new_quiv is not None:
                narrows = template["number of arrows"]
                UVC = [C[::narrows,::narrows] for C in new_quiv]
                quivs[u].set_UVC(*UVC)
        fig.canvas.draw_idle()
    
    out = interactive_output(update, sliders)
    # Display everything
    display(VBox(list(sliders.values()) + [out]))
      
def plot_bands(
        eigva:xr.DataArray,
        dim:str,
        xmin:float = None, xmax:float = None,
        ymin:float = None, ymax:float = None,
        figkw:dict = {}, 
        **linekw,
    ):
    """The main function to plot the eigenvalues in an interactive manner. Plot each eigenvalue as a function of the parameter dimension 'dim'.

    Args:
        eigva (xr.DataArray): The DataArray containing the eigenvalues, must have at least the 'band' dimension.
        dim (str): The dimension to plot against.
        xmin (float, optional): Like xmin from plt.plot. Defaults to None.
        xmax (float, optional): Like xmax from plt.plot. Defaults to None.
        ymin (float, optional): Like ymin from plt.plot. Defaults to None.
        ymax (float, optional): Like ymax from plt.plot. Defaults to None.
        figkw (dict, optional): A dictionnary passed to the plt.subplots function. Defaults to {}.
        Additional parameters are passed as kwargs to plt.plot().
    """
    slider_dims = [d for d in eigva.dims if d not in [dim, 'band']]
    sliders = {}
    for d in slider_dims:
        values = eigva.coords[d]
        sliders[d] = FloatSlider(
            value = values[0],
            min = values.min(),
            max = values.max(),
            step = (values.max()-values.min())/100,
            description = d,
        )
    
    initial_sel = {dim:sliders[dim].value for dim in slider_dims}
    
    initial_band = eigva.sel(initial_sel)

    fig, ax = plt.subplots(**figkw)
    lines = []
    for i in eigva.band:
        line = ax.plot(eigva.coords[dim], initial_band.sel(band = i), **linekw)
        lines += line

    
    ax.set_xlim(left = xmin, right = xmax)
    ax.set_ylim(bottom = ymin, top = ymax)
    ax.set_xlabel(dim)
    ax.set_ylabel('Energy')
    
    def update(**kwargs):
        sel = {dim:kwargs[dim] for dim in sliders}
        new_bands = eigva.sel(sel, method = 'nearest')
        for i,b in enumerate(eigva.band):
            lines[i].set_ydata(new_bands.sel(band = b).data)
    
    
    out = interactive_output(update, sliders)
    # Display everything
    display(VBox(list(sliders.values()) + [out]))
