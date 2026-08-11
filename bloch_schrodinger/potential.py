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
from bloch_schrodinger.utils import create_sliders, create_cart_grid
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
        unitvecs: list[list[float]],
        resolution: tuple[int],
        v0: Union[int, float, complex, np.generic, xr.DataArray] = 100,
        dtype: Union[Type[int], Type[float], Type[complex], Type[np.generic]] = float,
        endpoint: bool = False,
    ):
        """Initialize a Potential object, which is a wrapper around a main xarray.DataArray. The dimension of the space described
        by the Potential depends on the number of unit vectors given.

        Args:
            unitvecs (list[list[float]]): The lattice vectors of the created unit cell. The unit cell center is placed at (0,0)
            and the unit cell has sides a1 and a2.
            resolution (tuple[int,int]): The mesh resolution along a1 and a2.
            v0 (Union[int, float, complex, np.generic,xr.DataArray], optional): The Potential initial value, can be a single value of a parameter array (wrapped in DataArray).
            Defaults to 0.
            dtype (Union[Type[int],Type[float],Type[complex],Type[np.generic]], optional): The Potential type. Defaults to float.
            endpoint (bool, optional): Wheter to add the endpoint to a1 and a2 linspaces, important to ensure no double-counting of points when solving periodic problems. Defaults to False.

        Raises:
            ValueError: Raises an error if the unit vectors given don't have the proper shape.
        """

        # Checking the number of dimensions and consistency of inputs
        self.unitvecs = np.array(unitvecs)
        self.n_dims = np.shape(unitvecs)[0]

        if np.shape(unitvecs)[1] != self.n_dims:
            raise ValueError(
                "The length of each unit vector much match the number of unit vectors"
            )

        if self.n_dims != len(resolution):
            raise ValueError("unitvecs and resolution not of the same length")

        if np.isclose(self.get_surface(), 0):
            raise ValueError("The unit vectors must not be colinear")

        self.a = np.array(unitvecs)

        self.resolution = resolution
        self.n_elems = 1  # Total number of grid points
        for i in range(self.n_dims):
            self.n_elems *= self.resolution[i]

        self.v0 = v0
        self.dtype = dtype
        self.endpoint = endpoint

        V = np.ones(resolution, dtype=dtype)

        # The Potential landscapes are saved into a single DataArray
        self.V = (
            xr.DataArray(
                V,
                coords={
                    f"a{i + 1}": np.linspace(
                        -0.5, 0.5, resolution[i], endpoint=endpoint
                    )
                    + 1 / resolution[i] / 2 * (1 - endpoint)
                    for i in range(self.n_dims)
                },
            )
            * v0
        )

        # The multidimensional coordinates are made directly accessible to the potential object, just for convenience
        self.coord_names = ["x", "y", "z"]
        tmp_coords = [0] * 3
        self.coords = [0] * 3
        self.da = [0] * 3

        for i in range(self.n_dims):
            for j in range(self.n_dims):
                tmp_coords[i] = (
                    tmp_coords[i] + self.a[j, i] * self.V.coords[f"a{j + 1}"]
                )

        for i in range(self.n_dims):
            self.coords[i] = tmp_coords[i].assign_coords(
                {
                    self.coord_names[i]: tmp_coords[i],
                }
            )

            self.da[i] = (  # Length increment along ai
                float(
                    abs(self.V.coords[f"a{i + 1}"][1] - self.V.coords[f"a{i + 1}"][0])
                )
                * (self.a[i] @ self.a[i]) ** 0.5
            )

        # They are also stored directly into the DataArray
        self.V = self.V.assign_coords(
            {self.coord_names[i]: self.coords[i] for i in range(self.n_dims)}
        )

        # Definitions for retrocompatibility
        if self.n_dims == 2:
            self.da1 = self.da[0]
            self.da2 = self.da[1]
            self.a1 = self.a[0]
            self.a2 = self.a[1]
            self.x = self.coords[0]
            self.y = self.coords[1]

    def clear(self):
        """Remove all parameter dimensions and features from the potential"""
        V = np.ones(self.resolution, dtype=self.dtype)

        self.V = xr.DataArray(
            V * self.v0,
            coords={
                f"a{i + 1}": np.linspace(-0.5, 0.5, self.resolution[i], endpoint=self.endpoint)
                + 1 / self.resolution[i] / 2 * (1 - self.endpoint)
                for i in range(self.n_dims)
            },
        )

        for i in range(self.n_dims):
            for j in range(self.n_dims):
                self.coords[i] += self.a[j] * self.V.coords[f"a{j + 1}"]

            self.coords[i] = self.coords[i].assign_coords(
                {
                    self.coord_names[i]: self.coords[i],
                }
            )

            self.da[i] = (  # Length increment along ai
                float(
                    abs(self.V.coords[f"a{i + 1}"][1] - self.V.coords[f"a{i + 1}"][0])
                )
                * (self.a[i] @ self.a[i]) ** 0.5
            )

            # They are also stored directly into the DataArray
            self.V = self.V.assign_coords(
                {self.coord_names[i]: self.coords[i] for i in range(self.n_dims)}
            )

            # Definitions for retrocompatibility
            if self.n_dims == 2:
                self.da1 = self.da[0]
                self.da2 = self.da[1]
                self.a1 = self.a[0]
                self.a2 = self.a[1]

    # def __repr__(self) -> str:
    #     shape = {dim: len(self.V.coords[dim].data) for dim in self.V.dims}
    #     return f"Potential: \n a1 = {self.a1}, a2 = {self.a2} \n dimensions: {shape}"

    def add(self, value: xr.DataArray):
        """Changes the value of the potential everywhere by adding "value" to it

        Args:
            value (xr.DataArray): the DataArray to add to the potential, must be able to be broadcasted on self.V
        """
        self.V = self.V + value

    def get_surface(self) -> float:
        """Return the length/surface/volume of a potential object.

        Returns:
            float
        """

        return (np.abs(np.linalg.det(self.unitvecs))).item()

    def get_dS(self) -> float:
        """Return the length/surface/volume element dS of a potential object.

        Returns:
            float
        """
        return self.get_surface() / self.n_elems

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
            {self.coord_names[i]: self.coords[i] for i in range(self.n_dims)}
        )

    def circle(
        self,
        center: tuple[Union[float, xr.DataArray]],
        radius: Union[float, xr.DataArray],
        method: str = "set",
        inverse: bool = False,
        value: Union[float, xr.DataArray] = 0,
    ):
        """Change the value of the potential in a n-sphere. Support coordinates attribution for all parameters.

        Args:
            center (tuple[Union[float,xr.DataArray]]): The center of the n-sphere in the cartesian basis.
            radius (Union[float,xr.DataArray]): The radius of the circle
            method (str, optional): Wheter to replace the potential inside (method 'set') or to add the value to the potential in the circle (method 'add'). Defaults to 'set'.
            inverse (bool, optional): Wheter to replace the potential inside (False) or outside the rectangle.
            value (Union[float,xr.DataArray], optional): The value to set for the potential inside the circle. Defaults to 0.

        Raises:
            ValueError: If the method is not 'add' or 'set'
        """
        r = 0
        for i in range(self.n_dims):
            r = r + (self.coords[i] - center[i]) ** 2
        r = r**0.5

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

    def rotate_center(
        self,
        center: tuple[Union[float, xr.DataArray]],
        rotation: Union[float, xr.DataArray] = (0),
    )-> list[xr.DataArray]:
        """A small helper function to redefine the origin of coords and rotate them. Useful for drawing functions like rectangle or ellispe.
        center (tuple[Union[float,xr.DataArray]]): The center of the rectangle in the cartesian basis.
        dims (tuple[Union[float,xr.DataArray]]): The length along x and y.
        rotation (tuple[Union[float,xr.DataArray]]): A rotation (in radians) of the prism, as a tuple of angle. 
        First angle is for rotation around z, second for rotation around y. default to (0,0).
        """
        
        coord = [self.coords[i] - center[i] for i in range(self.n_dims)] 
        if self.n_dims == 1:
            return self.coords

        elif self.n_dims == 2:
            coord_rot = [
                coord[0] * xr.ufuncs.cos(rotation[0]) + coord[1] * xr.ufuncs.sin(rotation[0]),
                coord[0] * xr.ufuncs.sin(rotation[0]) - coord[1] * xr.ufuncs.cos(rotation[0]),
            ]
            return coord_rot

        else:
            coord_rot1 = [
                coord[0] * xr.ufuncs.cos(rotation[0]) + coord[1] * xr.ufuncs.sin(rotation[0]),
                coord[0] * xr.ufuncs.sin(rotation[0]) - coord[1] * xr.ufuncs.cos(rotation[0]),
                coord[2]
            ]

            coord_rot2 = [
                coord_rot1[0] * xr.ufuncs.cos(rotation[1]) + coord_rot1[2] * xr.ufuncs.sin(rotation[1]),
                coord_rot1[1],
                coord_rot1[0] * xr.ufuncs.sin(rotation[1]) - coord_rot1[2] * xr.ufuncs.cos(rotation[1]),
            ]
            return coord_rot2

    def rectangle(
        self,
        center: tuple[Union[float, xr.DataArray]],
        dims: tuple[Union[float, xr.DataArray]],
        rotation: tuple[Union[float, xr.DataArray]] = (0),
        method: str = "set",
        inverse: bool = False,
        value: Union[float, xr.DataArray] = 0,
    ):
        """Change the value of the potential in a rectangular prism. Support coordinates attribution for all parameters.

        Args:
            center (tuple[Union[float,xr.DataArray]]): The center of the rectangle in the cartesian basis.
            dims (tuple[Union[float,xr.DataArray]]): The length along x and y.
            rotation (tuple[Union[float,xr.DataArray]]): A rotation (in radians) of the prism, as a tuple of angle. 
            First angle is for rotation around z, second for rotation around y. default to (0,0).
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

        coords = self.rotate_center(center, rotation)

        if self.n_dims == 1:
            v1 = self.V * mod + value
            v2 = self.V

            if inverse:
                v1 = self.V
                v2 = self.V * mod + value

            self.V = xr.where(
                (abs(coords[0]) < dims[0] / 2), v1, v2
            )

        elif self.n_dims == 2:
            v1 = self.V * mod + value
            v2 = self.V

            if inverse:
                v1 = self.V
                v2 = self.V * mod + value

            self.V = xr.where(
                (abs(coords[0]) < dims[0] / 2) * (abs(coords[1]) < dims[1] / 2), v1, v2
            )

        else:            
            v1 = self.V * mod + value
            v2 = self.V

            if inverse:
                v1 = self.V
                v2 = self.V * mod + value

            self.V = xr.where(
                ((abs(coords[0]) < dims[0] / 2) * 
                 (abs(coords[1]) < dims[1] / 2) * 
                 (abs(coords[2]) < dims[2] / 2)
                 )
                , v1, v2
            )

    def ellipse(
        self,
        center: tuple[Union[float, xr.DataArray]],
        dims: tuple[Union[float, xr.DataArray]],
        rotation: tuple[Union[float, xr.DataArray]] = (0),
        method: str = "set",
        inverse: bool = False,
        value: Union[float, xr.DataArray] = 0,
    ):
        """Change the value of the potential in an ellipse. Support coordinates attribution for all parameters.

        Args:
            center (tuple[Union[float,xr.DataArray]]): The center of the ellipse in the x,y basis.
            dims (tuple[Union[float,xr.DataArray]]): The semi-axes along x and y.
            rotation (tuple[Union[float,xr.DataArray]]): A rotation (in radians) of the prism, as a tuple of angle. 
            First angle is for rotation around z, second for rotation around y. default to (0,0).
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

        coords = self.rotate_center(center, rotation)
        
        r = 0
        for i in range(self.n_dims):
            r = r + (coords[i] / dims[i]) ** 2
        r = r**0.5

        v1 = self.V * mod + value
        v2 = self.V

        if inverse:
            v1 = self.V
            v2 = self.V * mod + value

        self.V = xr.where(r < 1, v1, v2)

    def plot(
        self, cart_axes:list[int]=[0, 1], get_cbar=False, **kwargs
    ) -> Union[tuple[Figure, Axes], tuple[Figure, Axes, Axes]]:
        """Creates an interactive plot of the potential, with all the parameters as sliders.
        Must be used in an interactive python session, preferably. kwargs are passed to the
        matplotlib function used (plt.plot or plt.pcolormesh).

        Args:
            cart_axes (list[int], optional): The cartesian axes indexes against with to plot, with 0 = "x", 1 = "y" and 2 = "z".
            The number of axes given determine the plotting function used, currently only 1d and 2d plots are supported.Default to [0,1].
            get_cbar (bool, optional): Wheter to return the colorbar for modification. Defaults to False.
        """
        Vtmp = self.V.squeeze()
        self.V = self.V.assign_coords(
            {self.coord_names[i]: self.coords[i] for i in range(self.n_dims)}
        )
        
        ortho = cart_axes != [0, 1] or self.n_dims != 2
        if ortho:
            # Interpolate potential on a cartesian grid
            inv_coords = create_cart_grid(Vtmp)

            mapping = {f"a{i + 1}": inv_coords[i] for i in range(self.n_dims)}
            Vtmp = (
                Vtmp.interp(mapping, method="linear")
                .drop_vars([self.coord_names[i] for i in range(self.n_dims)])
                .rename(
                    {
                        self.coord_names[i] + "n": self.coord_names[i]
                        for i in range(self.n_dims)
                    }
                )
            )
            
            

        plot_dim = len(cart_axes)

        if plot_dim not in [1, 2]:
            raise ValueError("Too many or too little dimensions to plot against")

        slider_dims = [
            dim
            for dim in Vtmp.dims
            if dim not in [self.coord_names[j] for j in cart_axes]
        ] if ortho else [
            dim
            for dim in Vtmp.dims
            if dim not in ["a1", "a2"]
        ]
        
        sliders = create_sliders(Vtmp, slider_dims, start="mid")

        initial_sel = {dim: sliders[dim].value for dim in slider_dims}

        potential = Vtmp.sel(initial_sel, method="nearest")

        fig, ax = plt.subplots()
        if plot_dim == 1:
            co = Vtmp.coords[self.coord_names[cart_axes[0]]]
            obj = ax.plot(co, potential, **kwargs)
            ax.set_ylabel("Potential")

        else:
            tp = potential.transpose(
                self.coord_names[cart_axes[1]], self.coord_names[cart_axes[0]]
            ) if ortho else potential
            
            co1 = Vtmp.coords[self.coord_names[cart_axes[0]]]
            co2 = Vtmp.coords[self.coord_names[cart_axes[1]]]
            
            obj = ax.pcolormesh(co1, co2, tp, shading="auto", **kwargs)
            ax.set_ylabel(self.coord_names[cart_axes[1]])
            ax.set_aspect("equal")

        ax.set_xlabel(self.coord_names[cart_axes[0]])

        if plot_dim == 2:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(
                obj,
                cax=cax,
                label="Potential",
            )

            cbar.set_label("Potential")

        def update(**slkwargs):
            sel = {dim: slkwargs[dim] for dim in slider_dims}

            new_potential = Vtmp.sel(sel, method="nearest")

            if plot_dim == 1:
                pad = max(0.001,(new_potential.max() - new_potential.min()) * 0.05)
                obj[0].set_ydata(new_potential)
                ax.set_ylim(new_potential.min() - pad, new_potential.max() + pad)
            else:
                new_potential = new_potential.transpose(
                    self.coord_names[cart_axes[1]], self.coord_names[cart_axes[0]]
                ) if ortho else new_potential

                obj.set_array(new_potential.data.reshape(-1))
                obj.set_clim(
                    vmin=kwargs.get("vmin", float(new_potential.min())),
                    vmax=kwargs.get("vmax", float(new_potential.max())),
                )
                # ax.set_title(", ".join([f"{d}={sel[d]:.3f}" for d in sel]))
                fig.canvas.draw_idle()

        out = interactive_output(update, sliders)
        # Display everything
        display(VBox(list(sliders.values()) + [out]))
        if get_cbar:
            return fig, ax, cbar
        else:
            return fig, ax

    def copy(self):
        """Return a copy of the potential"""
        return deepcopy(self)

    def like(potential: "Potential") -> "Potential":
        """Returns a empty potential with the same specifications as the input potential

        Args:
            potential (Potential): the input potential whose parameters are to be copied

        Returns:
            Potential:
        """
        new_pot = potential.copy()
        new_pot.clear()
        return new_pot

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

    def coarsen(self, factor: tuple[int]) -> "Potential":
        """Return a coarsened copy of the Potential, with reduced resolution along all axes.

        Args:
            factor (tuple[int]): The coarsening factor. The initial resolution must be a multiple of the factor.

        Returns:
            Potential: Coarsened array
        """

        cpot = self.copy()
        arg = {f"a{i+1}":factor[i] for i in range(self.n_dims)}
        
        cpot.V = cpot.V.coarsen(arg).mean()
        cpot.resolution = (
            self.resolution[i] // factor[i] for i in range(self.n_dims)
        )

        cpot.coords[i] = [0]*self.n_dims
        tmp_coords = [0] * 3
        cpot.coords = [0] * 3
        cpot.da = [0] * 3

        for i in range(cpot.n_dims):
            for j in range(cpot.n_dims):
                tmp_coords[i] = (
                    tmp_coords[i] + cpot.a[j, i] * cpot.V.coords[f"a{j + 1}"]
                )

        for i in range(cpot.n_dims):
            cpot.coords[i] = tmp_coords[i].assign_coords(
                {
                    cpot.coord_names[i]: tmp_coords[i],
                }
            )

            cpot.da[i] = (  # Length increment along ai
                float(
                    abs(cpot.V.coords[f"a{i + 1}"][1] - cpot.V.coords[f"a{i + 1}"][0])
                )
                * (cpot.a[i] @ cpot.a[i]) ** 0.5
            )

        # They are also stored directly into the DataArray
        cpot.V = cpot.V.assign_coords(
            {cpot.coord_names[i]: cpot.coords[i] for i in range(cpot)}
        )

        # Definitions for retrocompatibility
        if cpot.n_dims == 2:
            cpot.da1 = cpot.da[0]
            cpot.da2 = cpot.da[1]
            cpot.a1 = cpot.a[0]
            cpot.a2 = cpot.a[1]
            cpot.x = cpot.coords[0]
            cpot.y = cpotelf.coords[1]

        return cpot

    def tile(self, bounds: tuple[tuple[int, int]]) -> "Potential":
        """Return a copy of the potential extended over multiple unit cells by tiling the original pattern.

        Args:
            bounds (tuple[tuple[int, int]]): A length-n_dims list of pairs of integers. Each pair describe the 
            number of unit cells to tile along the associated axe.

        Returns:
            Potential
        """

        non_spatial_coords = {
            dim: self.V.coords[dim] for dim in self.V.dims if dim not in [f"a{i+i}" for i in range(self.n_dims)]
        }

        reps = [bounds[i][1] - bounds[i][0] for i in range(self.n_dims)]
        
        ns = [self.V.sizes[f"a{i+1}"] for i in range(self.n_dims)]
        n_tots = [n * rep for n, rep in zip(reps, ns)]

        ls = [
            bounds[i][1] - 1 / 2 - (bounds[i][0] - 1 / 2) 
            for i in range(self.n_dims)
        ]

        non_spatial_coords.update(
            {
                f"a{i+1}": np.linspace(
                    bounds[i][0] - 1 / 2, bounds[i][1] - 1 / 2, n_tots[i], endpoint=False
                )
                + ls[i] / n_tots[i] / 2
                for i in range(self.n_dims)
            }
        )

        shape = tuple([coord.shape[0] for coord in non_spatial_coords.values()])

        new_V = xr.DataArray(np.zeros(shape, dtype=self.dtype), coords=non_spatial_coords)

        ncoords = [0] * 3

        for i in range(self.n_dims):
            for j in range(self.n_dims):
                ncoords[i] = (
                    ncoords[i] + self.a[j, i] * self.V.coords[f"a{j + 1}"]
                )

        new_V = new_V.assign_coords(
            {self.coord_names[i]:ncoords[i] for i in range(self.n_dims)}
        ).transpose(...,*[f"a{i+1}" for i in range(self.n_dims)])
        
        new_V.data = np.tile(
            self.V.transpose(...,*[f"a{i+1}" for i in range(self.n_dims)]), 
            reps=[1,*reps]
        )

        tiled = Potential(
            unitvecs=[self.a[i] * reps[i] for i in range(self.n_dims)],
            resolution=tuple(n_tots),
            v0=0,
            dtype=self.dtype,
        )
        tiled.V = new_V
        tiled.coords = [ncoords[i] for i in range(self.n_dims)]
        if self.n_dims == 2:
            tiled.da1 = ls[0]/n_tots[0]
            tiled.da2 = ls[1]/n_tots[1]
            tiled.a1 = tiled.a[0]
            tiled.a2 = tiled.a[1]
            tiled.x = tiled.coords[0]
            tiled.y = tiled.coords[1]

        return tiled


type paramType = Union[int, float, xr.DataArray]


def optical_honeycomb(
    wavelength: float,
    s1: Union[paramType, tuple[paramType, paramType, paramType]],
    s2: Union[paramType, tuple[paramType, paramType, paramType]],
    resolution: tuple[int, int] = (64, 64),
    theta: paramType = 0,
) -> tuple[dict, Potential]:
    """Construct a honeycomb lattice over a single unit cell.
    Args:
        potential (Potential): The target potential
        wavelength (float): wavelength of the lattice-generating lasers
        s1 (Union[paramType, tuple[paramType, paramType, paramType]]): Amplitude of the first triangular lattice. If a tuple of 3 is passed, each laser beam,s intensity is passed separately
        s2 (Union[paramType, tuple[paramType, paramType, paramType]]): Amplitude of the second triangular lattice. If a tuple of 3 is passed, each laser beam,s intensity is passed separately
        theta (paramType, optional): Rotates the whole lattice by an angle theta. Defaults to 0.

    Returns:
        tuple[dict, Potential]: A dictionnary containing most geometrical quntities relative to the lattice, and the constructed potential object
    """

    kl = 2 * np.pi / wavelength
    a = 4 * np.pi / 3**1.5 / kl

    M = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    a1 = M @ np.array([3 * a / 2, -(3**0.5) * a / 2])  # 1st lattice vector
    a2 = M @ np.array([3 * a / 2, 3**0.5 * a / 2])  # 2nd lattice vector

    E_r = kl**2 / 2
    # s0 = create_parameter('s0', np.linspace(3,10,3))

    k1 = kl * M @ np.array([-(3**0.5) / 2, 1 / 2])
    k2 = kl * M @ np.array([3**0.5 / 2, 1 / 2])
    k3 = kl * M @ np.array([0, -1])

    a1s = M @ np.array([-1, 3**0.5]) * 2 * np.pi / 3 / a
    a2s = M @ np.array([1, 3**0.5]) * 2 * np.pi / 3 / a
    K = M @ np.array([0, 4 * np.pi / 3**1.5 / a])

    V1 = -s2 * E_r
    V2 = -s1 * E_r

    honeycomb = Potential(
        unitvecs=[a1, a2],
        resolution=resolution,
        v0=0,
    )

    ### Lattice
    off = (a1 + a2) / 2
    dirs = [
        k1[0] * (honeycomb.x - off[0]) + k1[1] * (honeycomb.y - off[1]),
        k2[0] * (honeycomb.x - off[0]) + k2[1] * (honeycomb.y - off[1]),
        k3[0] * (honeycomb.x - off[0]) + k3[1] * (honeycomb.y - off[1]),
    ]

    tri_1 = 0
    tri_2 = 0
    for i in range(3):
        tri_1 += 2 * np.cos((dirs[i - 1] - dirs[i]) - 2 * np.pi / 3) / 2
        tri_2 += 2 * np.cos((dirs[i - 1] - dirs[i]) + 2 * np.pi / 3) / 2

    honeycomb.set(tri_1 * V1 + tri_2 * V2)

    constants = dict(
        a=a, E_r=E_r, a1=a1, a2=a2, a1s=a1s, a2s=a2s, K=K, k1=k1, k2=k2, k3=k3
    )

    return constants, honeycomb


def make_optical_honeycomb(
    potential: Potential,
    wavelength: float,
    s1: Union[paramType, tuple[paramType, paramType, paramType]],
    s2: Union[paramType, tuple[paramType, paramType, paramType]],
    theta: paramType = 0,
) -> tuple[dict, Potential]:
    """Construct a honeycomb lattice over the potential given

    Args:
        potential (Potential): The target potential
        wavelength (float): wavelength of the lattice-generating lasers
        s1 (Union[paramType, tuple[paramType, paramType, paramType]]): Amplitude of the first triangular lattice. If a tuple of 3 is passed, each laser beam,s intensity is passed separately
        s2 (Union[paramType, tuple[paramType, paramType, paramType]]): Amplitude of the second triangular lattice. If a tuple of 3 is passed, each laser beam,s intensity is passed separately
        theta (paramType, optional): Rotates the whole lattice by an angle theta. Defaults to 0.

    Returns:
        tuple[dict, Potential]: A dictionnary containing most geometrical quntities relative to the lattice, and the constructed potential object
    """

    kl = 2 * np.pi / wavelength
    a = 4 * np.pi / 3**1.5 / kl  # Lattice constant

    M = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    a1 = M @ np.array([3 * a / 2, -(3**0.5) * a / 2])  # 1st lattice vector
    a2 = M @ np.array([3 * a / 2, 3**0.5 * a / 2])  # 2nd lattice vector

    E_r = kl**2 / 2  # recoil energy
    # s0 = create_parameter('s0', np.linspace(3,10,3))

    k1 = kl * M @ np.array([-(3**0.5) / 2, 1 / 2])
    k2 = kl * M @ np.array([3**0.5 / 2, 1 / 2])
    k3 = kl * M @ np.array([0, -1])

    a1s = M @ np.array([-1, 3**0.5]) * 2 * np.pi / 3 / a  # reciprocal vector
    a2s = M @ np.array([1, 3**0.5]) * 2 * np.pi / 3 / a  # reciprocal vector
    K = M @ np.array([0, 4 * np.pi / 3**1.5 / a])  # BZ corner

    V1 = -s2 * E_r
    V2 = -s1 * E_r

    ### Lattice
    off = (a1 + a2) / 2
    dirs = [
        k1[0] * (potential.x - off[0]) + k1[1] * (potential.y - off[1]),
        k2[0] * (potential.x - off[0]) + k2[1] * (potential.y - off[1]),
        k3[0] * (potential.x - off[0]) + k3[1] * (potential.y - off[1]),
    ]

    tri_1 = 0
    tri_2 = 0
    for i in range(3):
        tri_1 += 2 * np.cos((dirs[i - 1] - dirs[i]) - 2 * np.pi / 3) / 2
        tri_2 += 2 * np.cos((dirs[i - 1] - dirs[i]) + 2 * np.pi / 3) / 2

    potential.set(tri_1 * V1 + tri_2 * V2)

    constants = dict(
        a=a, E_r=E_r, a1=a1, a2=a2, a1s=a1s, a2s=a2s, K=K, k1=k1, k2=k2, k3=k3
    )

    return constants, potential


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

    foo = Potential(
        [[10, 2, 0], [0, 5, 1], [1,2,8]],
        resolution=(100, 50, 70),
        endpoint=True,
        v0=0,
    )

    # foo = Potential(
    #     [[10, 2], [0, 5]],
    #     resolution=(100, 50),
    #     endpoint=True,
    #     v0=0,
    # )

    bar = create_parameter("bar", np.linspace(0,1,20))

    # foo.circle((0, 0, 2), bar+1, value=1)
    foo.ellipse((0, 0, 0), [4,2,2], rotation = [bar,2],value=1)
    foot = foo.tile([[-1,2],[0,1],[-2,1]])
    # dimp = create_parameter("dimp", np.linspace(0,1,50))
    # foo.set(foo.coords[0]**4)
    # foo.add((100-20*foo.coords[0]**2)*dimp)
    
    foot.plot(cart_axes=[1,0])
