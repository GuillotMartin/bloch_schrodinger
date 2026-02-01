import numpy as np
import xarray as xr
from typing import Union, Type
import matplotlib.pyplot as plt
from ipywidgets import VBox, interactive_output
from IPython.display import display
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from bloch_schrodinger.utils import create_sliders
from scipy.ndimage import gaussian_filter

scalar = [int, float, complex, np.generic]


def create_parameter(name: str, data: Union[list, np.ndarray]) -> xr.DataArray:
    """Create a DataArray containing a 1D coordinate, used to build a parameter space.
    Basically a dumbed-down container for DataArrays

    Args:
        name (str): Name of the parameter
        data (Union[list,np.ndarray]): The values taken by the parameters

    Returns:
        xr.DataArray: The resulting parameter wrapped in a DataArray.
    """
    arr = xr.DataArray(
        data=np.array(data),
        coords={name: np.array(data)},
        name=name,
    )

    return arr


class Potential:
    """The Potential class provides methods to create and modify energy potentials."""

    def __init__(
        self,
        unitvecs: list[list[float, float]],
        resolution: tuple[int, int],
        v0: Union[int, float, complex, np.generic, xr.DataArray] = 100,
        dtype: Union[Type[int], Type[float], Type[complex], Type[np.generic]] = float,
        endpoint: bool = False,
    ):
        """Initialize a Potential object.

        Args:
            unitvecs (list[list[float,float]]): The lattice vectors of the created unit cell. The unit cell center is placed at (0,0) and the unit cell has sides a1 and a2.
            resolution (tuple[int,int]): The mesh resolution along a1 and a2.
            v0 (Union[int, float, complex, np.generic,xr.DataArray], optional): The Potential initial value, can be a single value of a parameter array (wrapped in DataArray).
            Defaults to 0.
            dtype (Union[Type[int],Type[float],Type[complex],Type[np.generic]], optional): The Potential type. Defaults to float.
            dtype (bool, optional): Wheter to add the endpoint to a1 and a2 linspaces, important to ensure no double-counting of points when solving periodic problems. Defaults to False.

        Raises:
            ValueError: Raises an error if the unit vectors given don't have the proper shape.
        """
        self.a1 = np.array(unitvecs[0])
        self.a2 = np.array(unitvecs[1])

        self.v0 = v0
        self.resolution = resolution
        self.dtype = dtype

        if len(unitvecs) != 2:
            raise ValueError("Only two unit cell vectors must be given")
        elif len(self.a1) != 2:
            raise ValueError("first unit cell vector not of length 2")
        elif len(self.a2) != 2:
            raise ValueError("Second unit cell vector not of length 2")

        V = np.ones(resolution, dtype=dtype)

        # The Potential landscapes are saved into a single DataArray
        self.V = (
            xr.DataArray(
                V,
                coords={
                    "a1": np.linspace(
                        -0.5, 0.5, resolution[0], endpoint=endpoint
                    ),  # + 1/resolution[0]/2 * (1-endpoint),
                    "a2": np.linspace(
                        -0.5, 0.5, resolution[1], endpoint=endpoint
                    ),  # + 1/resolution[1]/2 * (1-endpoint),
                },
            )
            * v0
        )

        # The multidimensional coordinates x and y are made directly accessible to the potential object, just for convenience
        self.x = self.a1[0] * self.V.a1 + self.a2[0] * self.V.a2
        self.y = self.a1[1] * self.V.a1 + self.a2[1] * self.V.a2

        self.x = self.x.assign_coords(
            {
                "x": self.x,
                "y": self.y,
            }
        )
        self.y = self.y.assign_coords(
            {
                "x": self.x,
                "y": self.y,
            }
        )

        # They are also stored directly into the DataArray
        self.V = self.V.assign_coords(
            {
                "x": self.x,
                "y": self.y,
            }
        )

        self.da1 = (
            float(abs(self.V.a1[1] - self.V.a1[0])) * (self.a1 @ self.a1) ** 0.5
        )  # smallest increment of length along a1
        self.da2 = (
            float(abs(self.V.a2[1] - self.V.a2[0])) * (self.a2 @ self.a2) ** 0.5
        )  # smallest increment of length along a2

    def clear(self):
        """Remove all parameter dimensions and features from the potential"""
        V = np.ones(self.resolution, dtype=self.dtype)

        self.V = (
            xr.DataArray(
                V,
                coords={
                    "a1": np.linspace(-0.5, 0.5, self.resolution[0]),
                    "a2": np.linspace(-0.5, 0.5, self.resolution[1]),
                },
            )
            * self.v0
        )

        x = self.a1[0] * self.V.a1 + self.a2[0] * self.V.a2
        y = self.a1[1] * self.V.a1 + self.a2[1] * self.V.a2

        self.V = self.V.assign_coords(
            {
                "x": x,
                "y": y,
            }
        )

    def __repr__(self) -> str:
        shape = {dim: len(self.V.coords[dim].data) for dim in self.V.dims}
        return f"Potential: \n a1 = {self.a1}, a2 = {self.a2} \n dimensions: {shape}"

    def add(self, value: xr.DataArray):
        """Changes the value of the potential everywhere by adding "value" to it

        Args:
            value (xr.DataArray): the DataArray to add to the potential, must be able to be broadcasted on self.V
        """
        self.V = self.V + value

    def multiply(self, fac: xr.DataArray):
        """Changes the value of the potential everywhere by multiplying "fac" to it

        Args:
            fac (xr.DataArray): the DataArray to multiply to the potential, must be able to be broadcasted on self.V
        """
        self.V = self.V * fac

    def set(self, value: xr.DataArray):
        """Changes the value of the potential everywhere by setting it to "value"

        Args:
            value (xr.DataArray): the DataArray to set, must be able to be broadcasted on self.V
        """
        Vtmp = self.V * 1
        self.V = value + self.V - Vtmp
        self.V = self.V.assign_coords(
            {
                "x": self.x,
                "y": self.y,
            }
        )

    def circle(
        self,
        center: tuple[Union[float, xr.DataArray]],
        radius: Union[float, xr.DataArray],
        method: str = "set",
        inverse: bool = False,
        value: Union[float, xr.DataArray] = 0,
    ):
        """Change the value of the potential in a circle. Support coordinates attribution for all parameters.

        Args:
            center (tuple[Union[float,xr.DataArray]]): The center of the circle in the x,y basis.
            radius (Union[float,xr.DataArray]): The radius of the circle
            method (str, optional): Wheter to replace the potential inside (method 'set') or to add the value to the potential in the circle (method 'add'). Defaults to 'set'.
            inverse (bool, optional): Wheter to replace the potential inside (False) or outside the rectangle.
            value (Union[float,xr.DataArray], optional): The value to set for the potential inside the circle. Defaults to 0.

        Raises:
            ValueError: If the method is not 'add' or 'set'
        """
        r = ((self.V.x - center[0]) ** 2 + (self.V.y - center[1]) ** 2) ** 0.5
        if method == "set":
            mod = 0
        elif method == "add":
            mod = 1
        else:
            raise ValueError("method must be either 'set' or 'add'")

        v1 = self.V * mod + value
        v2 = self.V

        if inverse:
            v1 = self.V
            v2 = self.V * mod + value

        self.V = xr.where(r < radius, v1, v2)

    def rectangle(
        self,
        center: tuple[Union[float, xr.DataArray]],
        dims: tuple[Union[float, xr.DataArray]],
        rotation: Union[float, xr.DataArray] = 0,
        method: str = "set",
        inverse: bool = False,
        value: Union[float, xr.DataArray] = 0,
    ):
        """Change the value of the potential in a rectangle. Support coordinates attribution for all parameters.

        Args:
            center (tuple[Union[float,xr.DataArray]]): The center of the rectangle in the x,y basis.
            dims (tuple[Union[float,xr.DataArray]]): The length along x and y.
            rotation (Union[float,xr.DataArray]): A rotation (in radians) of the rectangle. default to 0.
            method (str, optional): Wheter to replace the potential inside (method 'set') or to add the value to the potential in the ellipse (method 'add'). Defaults to 'set'.
            inverse (bool, optional): Wheter to replace the potential inside (False) or outside the rectangle.
            value (Union[float,xr.DataArray], optional): The value to set for the potential inside the rectangle. Defaults to 0.

        Raises:
            ValueError: If the method is not 'add' or 'set'
        """
        if method == "set":
            mod = 0
        elif method == "add":
            mod = 1
        else:
            raise ValueError("method must be either 'set' or 'add'")

        x = self.V.x - center[0]
        y = self.V.y - center[1]

        x_rot = +x * xr.ufuncs.cos(rotation) + y * xr.ufuncs.sin(rotation)
        y_rot = +x * xr.ufuncs.sin(rotation) - y * xr.ufuncs.cos(rotation)

        v1 = self.V * mod + value
        v2 = self.V

        if inverse:
            v1 = self.V
            v2 = self.V * mod + value

        self.V = xr.where(
            (abs(x_rot) < dims[0] / 2) * (abs(y_rot) < dims[1] / 2), v1, v2
        )

    def ellipse(
        self,
        center: tuple[Union[float, xr.DataArray]],
        dims: tuple[Union[float, xr.DataArray]],
        rotation: Union[float, xr.DataArray] = 0,
        method: str = "set",
        inverse: bool = False,
        value: Union[float, xr.DataArray] = 0,
    ):
        """Change the value of the potential in an ellipse. Support coordinates attribution for all parameters.

        Args:
            center (tuple[Union[float,xr.DataArray]]): The center of the ellipse in the x,y basis.
            dims (tuple[Union[float,xr.DataArray]]): The semi-axes along x and y.
            rotation (Union[float,xr.DataArray]): A rotation (in radians) of the ellipse axis. default to 0.
            method (str, optional): Wheter to replace the potential inside (method 'set') or to add the value to the potential in the ellipse (method 'add'). Defaults to 'set'.
            inverse (bool, optional): Wheter to replace the potential inside (False) or outside the rectangle.
            value (Union[float,xr.DataArray], optional): The value to set for the potential inside the rectangle. Defaults to 0.

        Raises:
            ValueError: If the method is not 'add' or 'set'
        """

        if method == "set":
            mod = 0
        elif method == "add":
            mod = 1
        else:
            raise ValueError("method must be either 'set' or 'add'")

        x = self.V.x - center[0]
        y = self.V.y - center[1]

        x_rot = +x * xr.ufuncs.cos(rotation) + y * xr.ufuncs.sin(rotation)
        y_rot = +x * xr.ufuncs.sin(rotation) - y * xr.ufuncs.cos(rotation)

        r = ((x_rot / dims[0]) ** 2 + (y_rot / dims[1]) ** 2) ** 0.5

        v1 = self.V * mod + value
        v2 = self.V

        if inverse:
            v1 = self.V
            v2 = self.V * mod + value

        self.V = xr.where(r < 1, v1, v2)

    def plot(self, **kwargs) -> tuple[Figure, Axes]:
        """Creates an interactive plot of the potential, with all the parameters as sliders. Must be used in an interactive python session, preferably a notebook.
        kwargs are passed to the matplotlib pcolormesh function."""
        Vtmp = self.V.squeeze()
        self.V = self.V.assign_coords(
            {
                "x": self.x,
                "y": self.y,
            }
        )

        slider_dims = [dim for dim in Vtmp.dims if dim not in ["a1", "a2", "x", "y"]]
        sliders = create_sliders(Vtmp, slider_dims)

        initial_sel = {dim: sliders[dim].value for dim in slider_dims}
        potential = Vtmp.sel(initial_sel)

        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(Vtmp.x, Vtmp.y, potential, shading="auto", **kwargs)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(
            mesh,
            cax=cax,
            label="Potential",
        )

        cbar.set_label("Potential")

        def update(**slkwargs):
            sel = {dim: slkwargs[dim] for dim in slider_dims}

            new_potential = Vtmp.sel(sel, method="nearest")
            mesh.set_array(new_potential.data.reshape(-1))
            mesh.set_clim(
                vmin=kwargs.get("vmin", float(new_potential.min())),
                vmax=kwargs.get("vmax", float(new_potential.max())),
            )
            # ax.set_title(", ".join([f"{d}={sel[d]:.3f}" for d in sel]))
            fig.canvas.draw_idle()

        out = interactive_output(update, sliders)
        # Display everything
        display(VBox(list(sliders.values()) + [out]))
        return fig, ax

    def static_plot(self, selection: dict) -> tuple[Figure, Axes]:
        """A very simple static matplotlib function that should work everywhere matplotlib works.

        Args:
            selection (dict): A selection of all paramaters values. Cannot leave parameters out otherwise the potential won't be 2D

        Returns:
            Figure, Axes: The figure and its ax.
        """
        potential = self.V.sel(selection, method="nearest")
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(self.V.x, self.V.y, potential, shading="auto")
        ax.set_aspect("equal")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Potential")
        return fig, ax

    def copy(self):
        """Return a copy of the potential"""
        return deepcopy(self)

    def sel(self, selection: dict) -> "Potential":
        """Return a new potential with a subselection of the potential parameter space. See xarray 'sel' method for more infos.

        Args:
            selection (dict): _description_

        Returns:
            Potential: _description_
        """

        new_pot = self.copy()
        new_pot.V = new_pot.V.sel(selection)
        return new_pot

    def smooth(self, smooth_1: float, smooth_2: float, unit: str = "pixel"):
        """Smooth the potential by applying a gaussian filter to it (see scipy.ndimage.gaussian_filter). The smoothing strength can be given in pixels or units of a1 and a2.

        Args:
            smooth_1 (float): the smoothing strength along the direction a1
            smooth_2 (float): the smoothing strength along the direction a2
            unit (str, optional): Defines the units to use for the smoothing strength. Can be either 'pixel' or 'space'. Defaults to 'pixel'.
        """
        if unit == "space":
            smooth_1 /= self.da1
            smooth_2 /= self.da2
        elif unit != "pixel":
            raise ValueError("'unit' must either be 'space' or 'pixel'")

        def filter(arr: xr.DataArray) -> xr.DataArray:
            return gaussian_filter(arr, sigma=(smooth_1, smooth_2))

        smoothed_V = xr.apply_ufunc(
            filter,
            self.V,
            input_core_dims=[
                ["a1", "a2"]
            ],  # Telling apply_ufunc not to broadcast on these dimensions
            output_core_dims=[["a1", "a2"]],
        )
        self.V = smoothed_V

    def coarsen(self, factor: tuple[int, int]) -> "Potential":
        """Return a coarsened version of the Potential, with reduced resolution along a1 and a2.

        Args:
            factor (tuple[int, int]): The coarsening factor. The initial resolution must be a multiple of the factor.

        Returns:
            Potential: Coarsened array
        """

        cpot = self.copy()

        cpot.V = cpot.V.coarsen(a1=factor[0], a2=factor[1]).mean()
        cpot.resolution = (
            self.resolution[0] // factor[0],
            self.resolution[1] // factor[1],
        )

        cpot.x = self.a1[0] * cpot.V.a1 + self.a2[0] * cpot.V.a2
        cpot.y = self.a1[1] * cpot.V.a1 + self.a2[1] * cpot.V.a2

        cpot.x = cpot.x.assign_coords(
            {
                "x": cpot.x,
                "y": cpot.y,
            }
        )
        cpot.y = cpot.y.assign_coords(
            {
                "x": cpot.x,
                "y": cpot.y,
            }
        )

        # They are also stored directly into the DataArray
        cpot.V = cpot.V.assign_coords(
            {
                "x": cpot.x,
                "y": cpot.y,
            }
        )

        cpot.da1 = (
            float(abs(cpot.V.a1[1] - cpot.V.a1[0])) * (self.a1 @ self.a1) ** 0.5
        )  # smallest increment of length along a1
        cpot.da2 = (
            float(abs(cpot.V.a2[1] - cpot.V.a2[0])) * (self.a2 @ self.a2) ** 0.5
        )  # smallest increment of length along a2

        return cpot

    def tile(self, bounds1: tuple[int, int], bounds2: tuple[int, int]) -> "Potential":
        
        
        
        coords = {dim:self.V.coords[dim] for dim in self.V.dims if dim not in ['a1', 'a2']}
        
        reps1 = (bounds1[1]-bounds1[0]) 
        reps2 = (bounds2[1]-bounds2[0]) 
        na1, na2 = self.V.sizes['a1'], self.V.sizes['a2']
        na1_tot = na1 * reps1
        na2_tot = na2 * reps2
        
        coords.update(
            {'a1':np.linspace(bounds1[0], bounds1[1], na1_tot),
             'a2':np.linspace(bounds2[0], bounds2[1], na2_tot)}
        )
        
        shape = tuple(
            [coord.shape[0] for coord in coords.values()]
        )
        
        new_V = xr.DataArray(
            np.zeros(shape, dtype = self.dtype),
            coords=coords
        )

        tot_x = self.a1[0] * new_V.a1 + self.a2[0] * new_V.a2
        tot_y = self.a1[1] * new_V.a1 + self.a2[1] * new_V.a2

        # Some assignements
        tot_x = tot_x.assign_coords(
            {
                "x": tot_x,
                "y": tot_y,
            }
        )
        tot_y = tot_y.assign_coords(
            {
                "x": tot_x,
                "y": tot_y,
            }
        )
        
        new_V = new_V.assign_coords(
            {"x": tot_x, "y": tot_y}
        )

        # Loop da loop (sorry)
        for ia1 in range(reps1):
            for ia2 in range(reps2):
                lcR = {
                    'a1':slice(ia1*na1, (ia1+1)*na1),
                    'a2':slice(ia2*na2, (ia2+1)*na2)
                }
                                
                new_V[lcR] += self.V.transpose(..., 'a1', 'a2').data


        tiled = Potential(
            unitvecs=[self.a1*reps1, self.a2*reps2],
            resolution=(na1_tot, na2_tot),
            v0 = 0,
            dtype = self.dtype
        )
        
        tiled.V = new_V
        tiled.x = tot_x
        tiled.y = tot_y
        
        return tiled


def honeycomb(
    a: float,
    rA: Union[float,],
    rB: Union[float, xr.DataArray],
    res: tuple[int, int] = (100, 100),
    **kwargs,
) -> Potential:
    """Create a simple honeycomb lattice unit cell.

    Args:
        a (float): Inter-pillar distance
        rA (Union[float,xr.DataArray]): First pillar radius
        rB (Union[float,xr.DataArray]): Second pillar radius
        res (tuple[int,int], optional): Mesh resolution. Defaults to (100,100).
        args: Arguments to pass to the potential constructor
    """

    a1 = np.array([-(3**0.5) / 2 * a, 3 / 2 * a])  # 1st lattice vector
    a2 = np.array([3**0.5 / 2 * a, 3 / 2 * a])  # 2nd lattice vector

    Honey = Potential([a1, a2], res, **kwargs)

    posA = np.array([0, a / 2])
    posB = np.array([0, -a / 2])
    ucs = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)]
    for uc in ucs:
        centerA = posA + a1 * uc[0] + a2 * uc[1]
        centerB = posB + a1 * uc[0] + a2 * uc[1]
        Honey.circle(centerA, rA, value=0)
        Honey.circle(centerB, rB, value=0)

    return Honey


if __name__ == "__main__":
    dr = create_parameter("dr", [0, 0.1])
    pot = honeycomb(2.4, 2.75 / 2 - dr, 2.75 / 2 + dr)

    pot_coarse = pot.coarsen((1, 1))

    tiled = pot_coarse.tile([-3,2], [-3,2])
    
    tiled.plot()
