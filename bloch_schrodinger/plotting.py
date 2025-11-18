import numpy as np
import xarray as xr
from types import NoneType
from typing import Union, Type
from ipywidgets import interact, FloatSlider, IntSlider, HBox, VBox, Layout, interactive_output
from IPython.display import display
import cmcrameri.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy

from bloch_schrodinger.potential import Potential
from bloch_schrodinger.utils import create_sliders

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
    
    contours = {
        "levels": 0, 
        "colors": 'white', 
        "linewidths": 1, 
        "linestyles": 'dashed',
    }
    
    if name == 'amplitude':
        temp = {
            "colormap": cm.oslo_r,
            "norm": lambda: colors.Normalize(), # using a factory function to avoid colormap sharing
            "colorbarticks":None,
            "autoscale": True,
            "pcolormeshkwargs":{},
            "contourkwargs":deepcopy(contours),
            "cbarkwargs":{
                "format": "{x:.1e}",
            },
            "quiverkwargs":deepcopy(quivers),
        }
        temp['contourkwargs']['colors'] = 'black'
        return temp

    if name == 'amplitude - log':
        temp = {
            "colormap": cm.oslo,
            "norm": lambda: colors.LogNorm(),
            "colorbarticks":None,
            "autoscale": True,
            "pcolormeshkwargs":{},
            "contourkwargs":deepcopy(contours),
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
            "autoscale": True,
            "pcolormeshkwargs":{},
            "contourkwargs":deepcopy(contours),
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
            "contourkwargs":deepcopy(contours),
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
    potentials:list[list[Potential]], 
    templates:list[list[Union[str,dict]]],
    titles: Union[NoneType, list[list[str]]] = None,
    quivers:Union[NoneType, list[list[Union[NoneType, tuple[xr.DataArray]]]]] = None
    ):
    """The main function to plot eigenvectors in a interactive manner.

    Args:
        plots (list[list[xr.DataArray]]): A list of list of eigenvectors xr.DataArrays, each DataArray will be plotted in a separate subplot, 
        in a grid-pattern determined by the structure of the list of lists.
        potentials (list[list[Potential]]): The potentials to be plotted as contour for each plot.
        templates (list[list[Union[str,dict]]]): A dictionnary containing all the instruction to define each subplot style. the templates can also be strings 
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
    if len(templates) != n_rows or len(potentials) != n_rows: raise ValueError("different shapes for plots and templates")
    if len(templates[0]) != n_cols or len(potentials[0]) != n_cols: raise ValueError("different shapes for plots and templates")
    for i in range(1, n_rows):
        if len(plots[i]) != n_cols: raise ValueError(f"Length of row {i} of 'plots' not consistent")
        if len(templates[i]) != n_cols: raise ValueError(f"Length of row {i} of 'templates' not consistent")
        if len(potentials[i]) != n_cols: raise ValueError(f"Length of row {i} of 'potentials' not consistent")
    
    if not quivers:
        quivers = [[None]*n_cols for u in range(n_rows)]
    if not titles:
        titles = [['']*n_cols for u in range(n_rows)]
        
    sliders = {}
      
    def create_axe(
            fig:Figure, 
            ax:Axes, 
            plot:xr.DataArray, 
            potential:Potential, 
            quiver:Union[NoneType, tuple[xr.DataArray]], 
            template:dict, 
            title:str
        )-> tuple[Axes, QuadMesh, QuadContourSet, Quiver, dict]:
        """A function to create a single subplot

        Args:
            fig (Figure): the global figure object
            ax (Axes): The ax in which to plot.
            plot (xr.DataArray): The DataArray to plot as a heatmap.
            potential (Potential): The Potential to plot as a contour plot.
            quiver (Union[NoneType, tuple[xr.DataArray]]): The DataArray tuple to plot as a quiver.
            template (Union[str,dict]): The template, either a string to call 'get_template' or a dictionary
            title (str): The title of the subplot.

        Returns:
            tuple[Axes, dict, QuadMesh, QuadContourSet, Quiver, dict]: The axes, objects plotted and a template in the dictionary format
        """
        slider_dims = [dim for dim in plot.dims if dim not in ['a1','a2','x','y']]
        sliders_ax = create_sliders(plot, slider_dims)

        initial_field_sel = {dim:sliders_ax[dim].value for dim in slider_dims}
        initial_potential_sel = {dim:sliders_ax[dim].value for dim in slider_dims if dim in potential.V.dims}
        
        
        field = plot.sel(initial_field_sel)
        pot = potential.V.sel(initial_potential_sel)

        norm = template['norm']() if callable(template['norm']) else template['norm']
        mesh = ax.pcolormesh(
            plot.x, plot.y, field, 
            shading='auto', 
            cmap = template['colormap'],
            norm = norm,
            **template["pcolormeshkwargs"],
        )
        
        if 'clim' in template:
            mesh.set_clim(template['clim'][0], template['clim'][1])
                
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
            
        return ax, mesh, contour, quiv, sliders_ax
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
            
            template = get_template(templates[i][j]) if type(templates[i][j])==str else deepcopy(templates[i][j])
            
            ax, mesh, contour, quiv, slider_ax = create_axe(
                fig,
                axes[i][j], 
                plots[i][j], 
                potentials[i][j],
                quivers[i][j],
                template,
                titles[i][j]
            )
            
            dict_templates += [template]
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
            if template.get("autoscale", False):
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
        linekws:Union[dict, list[dict]] = dict(),
        figkw:dict = {},
    ):
    """The main function to plot the eigenvalues in an interactive manner. Plot each eigenvalue as a function of the parameter dimension 'dim'.

    Args:
        eigva (xr.DataArray): The DataArray containing the eigenvalues, must have at least the 'band' dimension.
        dim (str): The dimension to plot against.
        xmin (float, optional): Like xmin from plt.plot. Defaults to None.
        xmax (float, optional): Like xmax from plt.plot. Defaults to None.
        ymin (float, optional): Like ymin from plt.plot. Defaults to None.
        ymax (float, optional): Like ymax from plt.plot. Defaults to None.
        linekws (dict or list[dict], optional): keywords arguments to be passed to plt.plot. If a list of dictionnaries are given, then the dictionary linekws[i%len(linekws)] 
        is used for the i-th band. Defaults to {}.
        figkw (dict, optional): A dictionnary passed to the plt.subplots function. Defaults to {}.
    """
    
    list_linekw = linekws if type(linekws) == list else [linekws]
    slider_dims = [d for d in eigva.dims if d not in [dim, 'band']]
    sliders = create_sliders(eigva, slider_dims)
    
    initial_sel = {dim:sliders[dim].value for dim in slider_dims}
    
    initial_band = eigva.sel(initial_sel)

    fig, ax = plt.subplots(**figkw)
    lines = []
    nbands = len(eigva.band)
    for i, b in enumerate(eigva.band):
        line = ax.plot(eigva.coords[dim], initial_band.sel(band = b), **list_linekw[i%len(list_linekw)])
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

def energy_levels(
    eigva:xr.DataArray, 
    eigve:xr.DataArray, 
    potential:Potential, 
    res:int = 100, 
    ymin:float = None,
    ymax:float = None,
    frac:float = 0.05):
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
        
    band_dims = [dim for dim in eigva.dims if not dim == 'band']
    potential_dims = [dim for dim in potential.V.dims if dim not in ['a1', 'a2', 'x', 'y']]
    
    sliders = {**create_sliders(eigva, band_dims), **create_sliders(potential.V, potential_dims)}
    
    initial_potential_sel = {dim:sliders[dim].value for dim in potential_dims}
    initial_eigve_sel = {dim:sliders[dim].value for dim in band_dims}
    initial_eigva_sel = {dim:sliders[dim].value for dim in band_dims}
    
    bound = float(((potential.x**2+potential.y**2)**0.5).max())
    cut_coord = np.linspace(-bound, bound, res)
    
    slider_y = FloatSlider(
        value = 0,
        min = cut_coord[0],
        max = cut_coord[-1],
        step = (cut_coord[-1]-cut_coord[0])//res,
        description = 'offset'
    )
    
    slider_rot = FloatSlider(
        value = 0,
        min = 0,
        max = np.pi*2,
        step = np.pi/100,
        description = 'cut rotation (rad)'
    )
    
    x_coord, y_coord = np.cos(slider_rot.value)*cut_coord, np.sin(slider_rot.value)*cut_coord + slider_y.value
    e1, e2 = potential.a1/(potential.a1@potential.a1), potential.a2/(potential.a2@potential.a2)
    
    a1_coord = xr.DataArray(x_coord*e1[0] + y_coord*e1[1], coords = {'z':cut_coord})
    a2_coord = xr.DataArray(x_coord*e2[0] + y_coord*e2[1], coords = {'z':cut_coord})
    
    initial_potential = potential.V.sel(initial_potential_sel)
    initial_eigva = eigva.sel(initial_eigva_sel)
    initial_eigve = eigve.sel(initial_eigve_sel)
    
    initial_potential_slice = initial_potential.interp(a1 = a1_coord, a2 = a2_coord, kwargs={"fill_value": potential.v0})
    initial_eigve_slice = initial_eigve.interp(a1 = a1_coord, a2 = a2_coord, kwargs={"fill_value": 0})
    
    potential_range = (initial_potential_slice.real.max() - initial_potential_slice.real.min())
    ymin = initial_potential_slice.real.min() if ymin is None else ymin
    ymax = initial_potential_slice.real.max() if ymin is None else ymax
    plot_range = ymax - ymin
    
    initial_eigve_slice = initial_eigve_slice / abs(eigve).max() * plot_range*frac + initial_eigva

        
    pad = potential_range*0.03
    eigve_lines = []

    fig, ax = plt.subplots()
    
    potential_line = ax.fill_between(
        initial_potential_slice.z, 
        initial_potential_slice.real, 
        initial_potential_slice.real.min()-pad, 
        ec = 'none', fc = 'k', alpha = 0.3
    )
    
    for b in eigva.band:
        line = ax.fill_between(
            initial_eigve_slice.z, 
            initial_eigve_slice.sel(band=b), 
            initial_eigva.sel(band = b), 
            alpha = 0.5
        )
        eigve_lines += [line] 
    
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(cut_coord.min(),cut_coord.max())

    def update(**kwargs):
        sliders_params = {dim:kwargs[dim] for dim in kwargs if dim not in ['y', 'rot']}
        slider_y = kwargs['y']  
        slider_rot = kwargs['rot']
        
        x_coord, y_coord = np.cos(slider_rot)*cut_coord, np.sin(slider_rot)*cut_coord + slider_y
        
        a1_coord = xr.DataArray(x_coord*e1[0] + y_coord*e1[1], coords = {'z':cut_coord})
        a2_coord = xr.DataArray(x_coord*e2[0] + y_coord*e2[1], coords = {'z':cut_coord})
        
        potential_sel = {dim:sliders_params[dim] for dim in potential_dims}
        eigva_sel = {dim:sliders_params[dim] for dim in band_dims}
        eigve_sel = {dim:sliders_params[dim] for dim in band_dims}

        new_potential = potential.V.sel(potential_sel, method = 'nearest')
        new_eigva = eigva.sel(eigve_sel, method = 'nearest')
        new_eigve = eigve.sel(eigva_sel, method = 'nearest')
        
        potential_slice = new_potential.interp(a1 = a1_coord, a2 = a2_coord, kwargs={"fill_value": potential.v0})
        eigve_slice = new_eigve.interp(a1 = a1_coord, a2 = a2_coord, kwargs={"fill_value": 0})
    
        eigve_slice = eigve_slice / abs(eigve).max() * plot_range*frac + new_eigva
        
        potential_line.set_data(
            potential_slice.z, 
            potential_slice.real, 
            potential.V.real.min()-pad
        )
        
        for i, b in enumerate(eigva.band):
            eigve_lines[i].set_data(
                eigve_slice.z, 
                eigve_slice.sel(band=b), 
                new_eigva.sel(band = b)
            )
            
    
    out = interactive_output(update, {**sliders, 'y':slider_y, 'rot':slider_rot})
    # Display everything
    display(VBox([HBox(list(sliders.values())), HBox([slider_y, slider_rot]) ,out]))

def dashboard(
    eigva:xr.DataArray,
    eigvadim:str,
    eigveplots:list[list[Union[NoneType,xr.DataArray]]], 
    potential:Potential, 
    template:Union[str,dict],
    titles: Union[NoneType, list[list[Union[NoneType,str]]]] = None,
    quivers:Union[NoneType, list[list[Union[NoneType, tuple[xr.DataArray]]]]] = None,
    eigvawidth:int = 0.3,
    figkw:dict = {},
    gskw:dict = {},
    spines:bool = True,
    linekws:Union[list[dict], dict] = {"color":"blue"},
    autoscale:bool = True
    ):
    """A high-level function to plot the eigenvalues and the eigenvectors at the same time. see docs\AtomicToMolecular.ipynb for an example.

    Args:
        eigva (xr.DataArray): The eigenvalue DataArray
        eigvadim (str): The dimension to plot the eigenvalues against
        eigveplots (list[list[Union[NoneType,xr.DataArray]]]): A matrix representing the plot structure. 
        The figure will consist of a panel showing the eigenvalues, and beside it an array of pcolormeshes with the structure specified by this matrix
        potential (Potential): The potential for the contour overlay, only one needs to be given.
        template (Union[str,dict]): A template to use for the colormesh, see doc of 'plot_eigenvector' and 'get_template' for more infos.
        titles (Union[NoneType, list[list[Union[NoneType,str]]]], optional): The titles, either a matrix with the same shape as plot_matrix, or None. Defaults to None.
        quivers (Union[NoneType, list[list[Union[NoneType, tuple[xr.DataArray]]]]], optional): A quiverplot to overlay, see 'plot_eigenvectors' for more infos. Defaults to None.
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
    
    list_linekw = linekws if type(linekws) == list else [linekws]
    if not quivers:
        quivers = [[None]*n_cols for u in range(n_rows)]
    if not titles:
        titles = [['']*n_cols for u in range(n_rows)]
    
    bands_dims = [dim for dim in eigva.dims if dim not in ['band', eigvadim]]
    sliders = create_sliders(eigva, bands_dims)
      
    def create_axe(
            fig:Figure, 
            ax:Axes, 
            plot:xr.DataArray, 
            potential:Potential, 
            quiver:Union[NoneType, tuple[xr.DataArray]], 
            template:dict, 
            title:str
        )-> tuple[Axes, QuadMesh, QuadContourSet, Quiver, dict]:
        """A function to create a single subplot

        Args:
            fig (Figure): the global figure object
            ax (Axes): The ax in which to plot.
            plot (xr.DataArray): The DataArray to plot as a heatmap.
            potential (Potential): The Potential to plot as a contour plot.
            quiver (Union[NoneType, tuple[xr.DataArray]]): The DataArray tuple to plot as a quiver.
            template (Union[str,dict]): The template, either a string to call 'get_template' or a dictionary
            title (str): The title of the subplot.

        Returns:
            tuple[Axes, dict, QuadMesh, QuadContourSet, Quiver, dict]: The axes, objects plotted and a template in the dictionary format
        """
        slider_dims = [dim for dim in plot.dims if dim not in ['a1','a2','x','y']]
        sliders_ax = create_sliders(plot, slider_dims)

        initial_field_sel = {dim:sliders_ax[dim].value for dim in slider_dims}
        initial_potential_sel = {dim:sliders_ax[dim].value for dim in slider_dims if dim in potential.V.dims}
        
        
        field = plot.sel(initial_field_sel)
        pot = potential.V.sel(initial_potential_sel)

        norm = template['norm']() if callable(template['norm']) else template['norm']
        mesh = ax.pcolormesh(
            plot.x, plot.y, field, 
            shading='auto', 
            cmap = template['colormap'],
            norm = norm,
            **template["pcolormeshkwargs"],
        )
        
        if 'clim' in template:
            mesh.set_clim(template['clim'][0], template['clim'][1])
                
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
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for sp in ['bottom', 'top', 'left', 'right']:
            ax.spines[sp].set_visible(spines)
        ax.set_title(title)
        
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = fig.colorbar(
        #     mesh, 
        #     cax=cax,
        #     **template['cbarkwargs'],
        # )
        # if template.get("cbartickslabel"):
        #     ticks = template["cbarkwargs"].get("ticks", cbar.ax.get_yticks())
        #     cbar.set_ticks(ticks)
        #     cbar.set_ticklabels(template["cbartickslabel"])
            
        return ax, mesh, contour, quiv, sliders_ax
        # cbar.set_label("Potential")
    
    
    fig = plt.figure(figsize=(min(3*(n_cols_tot + 1), 10), max(3*(n_rows-1), 3)), **figkw)
    gs_bands = GridSpec(1, 1, left=0.05, right=eigvawidth)
    gs_eigenvectors = GridSpec(n_rows, n_cols, 1, left=eigvawidth+0.05, right=0.98, **gskw)
    
    axes = []
    indx, jndx, meshes, contours, quivs, dict_templates = [], [], [], [], [], []
    
    for i in range(n_rows):
        for j in range(n_cols):
            if eigveplots[i][j] is not None:
                template = get_template(template) if type(template)==str else deepcopy(template)
                ax = fig.add_subplot(gs_eigenvectors[i, j])
                ax, mesh, contour, quiv, slider_ax = create_axe(
                    fig,
                    ax, 
                    eigveplots[i][j], 
                    potential,
                    quivers[i][j],
                    template,
                    titles[i][j]
                )
                
                dict_templates += [template]
                axes += [ax]
                indx += [i]
                jndx += [j]
                meshes += [mesh]
                contours += [contour]
                quivs += [quiv]
                sliders.update(slider_ax)
    
    ax = fig.add_subplot(gs_bands[0,0])
    initial_eigva_sel = {dim:sliders[dim].value for dim in bands_dims}
    initial_eigva = eigva.sel(initial_eigva_sel)
    
    lines = []
    for i, b in enumerate(eigva.band):
        line = ax.plot(eigva.coords[eigvadim], initial_eigva.sel(band = b), **list_linekw[i%len(list_linekw)])
        lines += line
    ax.set_xlabel(eigvadim)
    dim_pos = ax.axvline(sliders[eigvadim].value, linestyle = 'dashed', color = 'red')


    def update_eigenvectors(**kwargs):
        eigve_sel = {dim:kwargs[dim] for dim in sliders}
        new_plots, new_potentials, new_quivs = [], [], []
        for i in range(n_rows):
            for j in range(n_cols):
                if eigveplots[i][j] is not None:
                    field_sel = {dim:value for dim, value in eigve_sel.items() 
                                if dim in eigveplots[i][j].dims 
                                and dim not in ['a1','a2','x','y']}
                    
                    potential_sel = {dim:value for dim, value in field_sel.items() 
                                    if dim in potential.V.dims}
                    
                    new_plots += [eigveplots[i][j].sel(field_sel, method = 'nearest')]
                    new_potentials += [potential.V.sel(potential_sel, method = 'nearest')]
                    
                    if quivers[i][j] is not None:
                        new_quiv = [C.sel(field_sel, method = 'nearest') for C in quivers[i][j]]
                    else:
                        new_quiv = None
                    new_quivs += [new_quiv]
                

        for u, (new_plot, new_pot, new_quiv) in enumerate(zip(new_plots, new_potentials, new_quivs)):
            meshes[u].set_array(new_plot.data.reshape(-1))
            if template.get("autoscale", False):
                meshes[u].autoscale()

            contours[u].remove()
            
            contours[u] = axes[u].contour(
                new_plot.x, new_plot.y, new_pot, 
                **template['contourkwargs']
            )
            if new_quiv is not None:
                narrows = template["number of arrows"]
                UVC = [C[::narrows,::narrows] for C in new_quiv]
                quivs[u].set_UVC(*UVC)

    def update_bands(**kwargs):
        sel = {dim:kwargs[dim] for dim in bands_dims}
        new_bands = eigva.sel(sel, method = 'nearest')
        for i,b in enumerate(eigva.band):
            lines[i].set_ydata(new_bands.sel(band = b).data)
        dim_pos.set_xdata([kwargs[eigvadim], kwargs[eigvadim]])
        if autoscale:
            pad = (new_bands.max()-new_bands.min())*0.05
            ax.set_ylim(new_bands.min()-pad, new_bands.max()+pad)

    def update(**kwargs):
        update_eigenvectors(**kwargs)
        update_bands(**kwargs)
        fig.canvas.draw_idle()
    
    out = interactive_output(update, sliders)
    # Display everything
    display(VBox(list(sliders.values()) + [out]))

