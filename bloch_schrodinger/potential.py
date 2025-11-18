import numpy as np
import xarray as xr
from typing import Union, Type
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, IntSlider, VBox, interactive_output
from IPython.display import display
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from bloch_schrodinger.utils import create_sliders

scalar = [int, float, complex, np.generic]

def create_parameter(name:str, data:Union[list,np.ndarray])->xr.DataArray:
    """Create a DataArray containing a 1D coordinate, used to build a parameter space.
    Basically a dumbed-down container for DataArrays  

    Args:
        name (str): Name of the parameter
        data (Union[list,np.ndarray]): The values taken by the parameters

    Returns:
        xr.DataArray: The resulting parameter wrapped in a DataArray.
    """
    arr = xr.DataArray(
        data = np.array(data),
        coords = {name:np.array(data)},
        name = name,
    )
    
    return arr

class Potential:
    """The Potential class provides methods to create and modify energy potentials.
    """
    
    def __init__(
            self,
            unitvecs:list[list[float,float]], 
            resolution:tuple[int,int], 
            v0:Union[int, float, complex, np.generic,xr.DataArray] = 100, 
            dtype:Union[Type[int],Type[float],Type[complex],Type[np.generic]] = float
        ):
        """Initialize a Potential object.

        Args:
            unitvecs (list[list[float,float]]): The lattice vectors of the created unit cell. The unit cell center is placed at (0,0) and the unit cell has sides a1 and a2.
            resolution (tuple[int,int]): The mesh resolution along a1 and a2.
            v0 (Union[int, float, complex, np.generic,xr.DataArray], optional): The Potential initial value, can be a single value of a parameter array (wrapped in DataArray). 
            Defaults to 0.
            dtype (Union[Type[int],Type[float],Type[complex],Type[np.generic]], optional): The Potential type. Defaults to float.

        Raises:
            ValueError: Raises an error if the unit vectors given don't have the proper shape.
        """
        self.a1 = np.array(unitvecs[0])
        self.a2 = np.array(unitvecs[1])
        self.v0 = v0
        self.resolution = resolution
        self.dtype = dtype
        
        if len(unitvecs) != 2: raise ValueError("Only two unit cell vectors must be given")
        elif len(self.a1) != 2: raise ValueError("first unit cell vector not of length 2")
        elif len(self.a2) != 2: raise ValueError("Second unit cell vector not of length 2")
        
        V = np.ones(resolution, dtype = dtype)
        
        # The Potential landscapes are saved into a single DataArray
        self.V = xr.DataArray(
            V,
            coords={
                "a1":np.linspace(-0.5,0.5, resolution[0]),
                "a2":np.linspace(-0.5,0.5, resolution[1])
                }
            ) * v0
        
        # The multidimensional coordinates x and y are made directly accessible to the potential object, just for convenience
        self.x = self.a1[0]*self.V.a1 + self.a2[0]*self.V.a2
        self.y = self.a1[1]*self.V.a1 + self.a2[1]*self.V.a2

        # They are also stored directly into the DataArray
        self.V = self.V.assign_coords({
            "x":self.x,
            "y":self.y,
        })
        
    def clear(self):
        """Remove all parameter dimensions and features from the potential"""
        V = np.ones(self.resolution, dtype = self.dtype)
        
        self.V = xr.DataArray(
            V,
            coords={
                "a1":np.linspace(-0.5,0.5, self.resolution[0]),
                "a2":np.linspace(-0.5,0.5, self.resolution[1]),
            }
        ) * self.v0
        
        x = self.a1[0]*self.V.a1 + self.a2[0]*self.V.a2
        y = self.a1[1]*self.V.a1 + self.a2[1]*self.V.a2

        self.V = self.V.assign_coords({
            "x":x,
            "y":y,
        })
        
    def __repr__(self)->str:
        shape = {dim:len(self.V.coords[dim].data) for dim in self.V.dims}
        return f"Potential: \n a1 = {self.a1}, a2 = {self.a2} \n dimensions: {shape}"
    
    def circle(self, 
               center:tuple[Union[float,xr.DataArray]], 
               radius:Union[float,xr.DataArray],
               method:str = 'set',
               value:Union[float,xr.DataArray] = 0,
               ):
        """Change the value of the potential in a circle. Support coordinates attribution for all parameters.

        Args:
            center (tuple[Union[float,xr.DataArray]]): The center of the circle in the x,y basis.
            radius (Union[float,xr.DataArray]): The radius of the circle
            method (str, optional): Wheter to replace the potential inside (method 'set') or to add the value to the potential in the circle (method 'add'). Defaults to 'set'.
            value (Union[float,xr.DataArray], optional): The value to set for the potential inside the circle. Defaults to 0.

        Raises:
            ValueError: If the method is not 'add' or 'set'
        """
        r = ((self.V.x - center[0])**2 + (self.V.y - center[1])**2)**0.5
        
        if method == 'set':
            mod = 0
        elif method == 'add':
            mod = 1
        else: raise ValueError("method must be either 'set' or 'add'")
        
        self.V = xr.where(r < radius, self.V*mod + value, self.V)
        
        
    def rectangle(self, 
               center:tuple[Union[float,xr.DataArray]], 
               dims:tuple[Union[float,xr.DataArray]],
               rotation:Union[float,xr.DataArray] = 0,
               method:str = 'set',
               value:Union[float,xr.DataArray] = 0,
               ):
        
        """Change the value of the potential in a rectangle. Support coordinates attribution for all parameters.

        Args:
            center (tuple[Union[float,xr.DataArray]]): The center of the rectangle in the x,y basis.
            dims (tuple[Union[float,xr.DataArray]]): The length along x and y.
            rotation (Union[float,xr.DataArray]): A rotation (in radians) of the rectangle. default to 0.
            method (str, optional): Wheter to replace the potential inside (method 'set') or to add the value to the potential in the ellipse (method 'add'). Defaults to 'set'.
            value (Union[float,xr.DataArray], optional): The value to set for the potential inside the rectangle. Defaults to 0.

        Raises:
            ValueError: If the method is not 'add' or 'set'
        """
        if method == 'set':
            mod = 0
        elif method == 'add':
            mod = 1
        else: raise ValueError("method must be either 'set' or 'add'")

        x = self.V.x - center[0]
        y = self.V.y - center[1]
        
        x_rot = + x * xr.ufuncs.cos(rotation) + y * xr.ufuncs.sin(rotation)
        y_rot = + x * xr.ufuncs.sin(rotation) - y * xr.ufuncs.cos(rotation)
        
        self.V = xr.where((abs(x_rot) < dims[0]/2) * (abs(y_rot) < dims[1]/2), 
                          self.V*mod + value, 
                          self.V)

    def ellipse(self, 
            center:tuple[Union[float,xr.DataArray]], 
            dims:tuple[Union[float,xr.DataArray]],
            rotation:Union[float,xr.DataArray] = 0,
            method:str = 'set',
            value:Union[float,xr.DataArray] = 0,
            ):
        """Change the value of the potential in an ellipse. Support coordinates attribution for all parameters.

        Args:
            center (tuple[Union[float,xr.DataArray]]): The center of the ellipse in the x,y basis.
            dims (tuple[Union[float,xr.DataArray]]): The semi-axes along x and y.
            rotation (Union[float,xr.DataArray]): A rotation (in radians) of the ellipse axis. default to 0.
            method (str, optional): Wheter to replace the potential inside (method 'set') or to add the value to the potential in the ellipse (method 'add'). Defaults to 'set'.
            value (Union[float,xr.DataArray], optional): The value to set for the potential inside the rectangle. Defaults to 0.

        Raises:
            ValueError: If the method is not 'add' or 'set'
        """
        
        if method == 'set':
            mod = 0
        elif method == 'add':
            mod = 1
        else: raise ValueError("method must be either 'set' or 'add'")

        x = self.V.x - center[0]
        y = self.V.y - center[1]
        
        x_rot = + x * xr.ufuncs.cos(rotation) + y * xr.ufuncs.sin(rotation)
        y_rot = + x * xr.ufuncs.sin(rotation) - y * xr.ufuncs.cos(rotation)
        
        r = ((x_rot/dims[0])**2 + (y_rot/dims[1])**2)**0.5
        
        self.V = xr.where(r < 1, 
                        self.V*mod + value, 
                        self.V)

    def plot(self):
        """Creates an interactive plot of the potential, with all the parameters as sliders. Must be used in an interactive python session, preferably a notebook.
        """
        slider_dims = [dim for dim in self.V.dims if dim not in ['a1','a2','x','y']]
        sliders = create_sliders(self.V, slider_dims)

        initial_sel = {dim:sliders[dim].value for dim in slider_dims}
        potential = self.V.sel(initial_sel)

        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(self.V.x, self.V.y, potential, shading='auto')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(
            mesh, 
            cax=cax,
            label='Potential', 
        )        
        
        
        cbar.set_label("Potential")
        
        def update(**kwargs):
            sel = {dim:kwargs[dim] for dim in slider_dims}
            
            new_potential = self.V.sel(sel, method = 'nearest')
            mesh.set_array(new_potential.data.reshape(-1))
            mesh.set_clim(vmin=float(new_potential.min()), vmax=float(new_potential.max()))
            ax.set_title(", ".join([f"{d}={sel[d]:.3f}" for d in sel]))
            fig.canvas.draw_idle()
        
        out = interactive_output(update, sliders)
        # Display everything
        display(VBox(list(sliders.values()) + [out]))

    def static_plot(self, selection:dict)->tuple[Figure, Axes]:
        """A very simple static matplotlib function that should work everywhere matplotlib works.

        Args:
            selection (dict): A selection of all paramaters values. Cannot leave parameters out otherwise the potential won't be 2D

        Returns:
            Figure, Axes: The figure and its ax.
        """
        potential = self.V.sel(selection, method = 'nearest')
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(self.V.x, self.V.y, potential, shading='auto')
        ax.set_aspect('equal')
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Potential")
        return fig, ax
        
        
    def copy(self)->Potential:
        """Return a copy of the potential
        """
        return deepcopy(self)

def honeycomb(a:float, rA:Union[float,], rB:Union[float,xr.DataArray], res:tuple[int,int] = (100,100), **kwargs)->Potential:
    """Create a simple honeycomb lattice unit cell.

    Args:
        a (float): Inter-pillar distance
        rA (Union[float,xr.DataArray]): First pillar radius
        rB (Union[float,xr.DataArray]): Second pillar radius
        res (tuple[int,int], optional): Mesh resolution. Defaults to (100,100).
        args: Arguments to pass to the potential constructor
    """
    
    a1 = np.array([-3**0.5/2 * a, 3/2 * a]) # 1st lattice vector
    a2 = np.array([ 3**0.5/2 * a, 3/2 * a]) # 2nd lattice vector

    Honey = Potential([a1,a2], res,**kwargs)

    posA = np.array([0,a/2])
    posB = np.array([0,-a/2])
    ucs = [(0,0),(0,1),(0,-1),(-1,0),(1,0)]
    for uc in ucs:
        centerA = posA + a1*uc[0] + a2*uc[1]
        centerB = posB + a1*uc[0] + a2*uc[1]
        Honey.circle(centerA, rA, value = 0)
        Honey.circle(centerB, rB, value = 0)
    
    return Honey

