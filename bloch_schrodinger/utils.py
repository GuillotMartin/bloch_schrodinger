import numpy as np
import xarray as xr
from typing import Union
from ipywidgets import FloatSlider, IntSlider
# from bloch_schrodinger.potential import Potential
import matplotlib.pyplot as plt

### ====================
### Sliders
### ====================
def islinspace(arr: xr.DataArray) -> tuple[bool, float]:
    """Checks wether a coordinate array is regularily spaced, and return a boolean and the step size.

    Args:
        arr (xr.DataArray): The 1D array to check

    Returns:
        tuple[bool, float]: The first element is a boolean equal to True if it is regularily spaced. If True, then second element is the step size, otherwise it is 0.
    """

    step = 0
    islin = False
    steps = (arr.data - np.roll(arr.data, -1))[:-1]
    if np.all(np.isclose(steps, steps[0])):
        step = abs(steps[0])
        islin = True
    return islin, step


def create_sliders(
    arr: xr.DataArray, list_dims: list[str], start: str = "left"
) -> dict[Union[IntSlider, FloatSlider]]:
    """Create a dictionnary of sliders from the dimensions of a DataArray.

    Args:
        arr (xr.DataArray): The array whose coordinates are going to be used
        list_dims (list[str]): The list of dimensions keys for which to create sliders.
        start (str): The default position of sliders, either 'left', 'right' or 'mid'. default to left.

    Returns:
        dict[FloatSlider]: _description_
    """

    sliders_ax = {}
    for dim in list_dims:
        coord = arr.coords[dim]
        val = coord.min() if start == "left" else (coord.max() + coord.min()) / 2
        islin, step = islinspace(coord)
        if np.issubdtype(coord.dtype, np.floating):
            sliders_ax[dim] = FloatSlider(
                min=float(coord.min()),
                max=float(coord.max()),
                step=float((coord.max() - coord.min()) / max(100, len(coord)))
                if not islin
                else step,
                value=float(val),
                description=dim,
            )
        else:
            sliders_ax[dim] = IntSlider(
                min=coord.min(),
                max=coord.max(),
                step=1 if not islin else step,
                value=val,
                description=dim,
            )
    return sliders_ax


def create_sliders_from_dims(
    coordinates: dict[xr.DataArray], start: str = "left"
) -> dict[Union[IntSlider, FloatSlider]]:
    """Create a dictionnary of sliders from a dictionnary of dimensions.

    Args:
        coordinates (dict[xr.DataArray]): The dimensions for which to create the sliders.
        start (str): The default position of sliders, either 'left', 'right' or 'mid'. default to left.

    Returns:
        dict[FloatSlider]: _description_
    """

    sliders_ax = {}
    for dim, coord in coordinates.items():
        val = (
            coord.min().item()
            if start == "left"
            else (coord.max().item() - coord.min().item()) / 2
        )
        islin, step = islinspace(coord)
        if np.issubdtype(coord.dtype, np.floating):
            sliders_ax[dim] = FloatSlider(
                min=float(coord.min()),
                max=float(coord.max()),
                step=float((coord.max() - coord.min()) / max(100, len(coord)))
                if not islin
                else step,
                value=float(val),
                description=dim,
            )
        else:
            sliders_ax[dim] = IntSlider(
                min=coord.min(),
                max=coord.max(),
                step=1 if not islin else step,
                value=val,
                description=dim,
            )
    return sliders_ax


### ====================
### Arrays
### ====================


def coarsen(arr: xr.DataArray, factor: tuple[int, int]) -> xr.DataArray:
    """Return a coarsened version of the array, with reduced resolution along a1 and a2.

    Args:
        arr (xr.DataArray): The array to coarsen
        factor (tuple[int, int]): The coarsening factor. The initial resolution must be a multiple of the factor.

    Returns:
        Potential: Coarsened array
    """
    carr = arr.coarsen(a1=factor[0], a2=factor[1]).mean()
    return carr


def tile(
    arr: xr.DataArray,
    unitvecs: list[list[float, float]],
    bounds1: tuple[int, int],
    bounds2: tuple[int, int],
) -> xr.DataArray:
    """Return a copy of the array extended over multiple unit cells by tiling the original pattern.

    Args:
        arr (xr.DataArray): The array to tile
        unitvecs (list[list[float, float]]): The unit vectors to use for tiling.
        bounds1 (tuple[int, int]): Numbers of cells along a1
        bounds2 (tuple[int, int]): Number of cells along a2

    Returns:
        xr.DataArray
    """

    coords = {dim: arr.coords[dim] for dim in arr.dims if dim not in ["a1", "a2"]}

    reps1 = bounds1[1] - bounds1[0]
    reps2 = bounds2[1] - bounds2[0]
    na1, na2 = arr.sizes["a1"], arr.sizes["a2"]
    na1_tot = na1 * reps1
    na2_tot = na2 * reps2

    coords.update(
        {
            "a1": np.linspace(
                bounds1[0] - 1 / 2, bounds1[1] - 1 / 2, na1_tot, endpoint=False
            ),
            "a2": np.linspace(
                bounds2[0] - 1 / 2, bounds2[1] - 1 / 2, na2_tot, endpoint=False
            ),
        }
    )

    shape = tuple([coord.shape[0] for coord in coords.values()])

    new_arr = xr.DataArray(np.zeros(shape, dtype=arr.dtype), coords=coords)

    a1 = unitvecs[0]
    a2 = unitvecs[1]

    tot_x = a1[0] * new_arr.a1 + a2[0] * new_arr.a2
    tot_y = a1[1] * new_arr.a1 + a2[1] * new_arr.a2

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

    new_arr = new_arr.assign_coords({"x": tot_x, "y": tot_y})

    # Loop da loop (sorry)
    for ia1 in range(reps1):
        for ia2 in range(reps2):
            lcR = {
                "a1": slice(ia1 * na1, (ia1 + 1) * na1),
                "a2": slice(ia2 * na2, (ia2 + 1) * na2),
            }

            new_arr[lcR] += arr.transpose(..., "a1", "a2").data

    return new_arr


def create_cart_grid(
    arr: xr.DataArray, resolution: Union[int, tuple[int]] = 100
) -> tuple[xr.DataArray]:
    """Create a cartesian grid describing a bounding box for a real-field array (like a Potential.V or a wavefunction array).
    This function's main goal is to be used as interpolation grids for plotting.

    Args:
        arr (xr.DataArray): The array to create a bounding box for.
        resolution (Union[int, tuple[int]], optional): The resolution for the cartesian coordinates. If a single int is given, 
        this will be the resolution of the coordinate with the longest extend. If a tuple is given, it much match the dimnesionality of the array, 
        and each cartesian coordinates will be given the specified resolution. Default to 100.
    Returns:
        tuple[xr.DataArray]: Three arrays "x", "y", "z" with dimensions "ai".
    """

    n_dims = len(
        [dim for dim in arr.dims if dim in ["a1", "a2", "a3"]]
    )  # Determines the cartesian dimension of the array
    name_coords = ["x", "y", "z"]  # Cartesian coordinates names

    a = np.zeros((n_dims, n_dims))  # Reconstructing the gen,erating unit vectors

    origin = [0] * n_dims
    for i in range(n_dims):
        end = [0] * n_dims
        end[i] = -1
        for j in range(n_dims):
            a[i, j] = arr.coords[name_coords[j]][*end] - arr.coords[name_coords[j]][*origin]

    # Creating the bounding box
    if n_dims == 3:
        perms = np.array([
            [1,1,1], [1,1,-1], [1,-1,1], [-1,1,1], 
            [-1,-1,1], [-1,1,-1], [1,-1,-1], [-1,-1,-1]
        ])
    elif n_dims == 2:
        perms = np.array([
            [1,1], [1,-1], [-1,1], [-1,-1], 
        ])
    else:
        perms = np.array([
            [1],[-1], 
        ])
    
    vertices = perms@a/2
    mins  = np.min(vertices, axis=0)
    maxs  = np.max(vertices, axis=0)
    rng = maxs - mins
    ## Setting  the resolution
    if isinstance(resolution, int):
        tmp_resolution = [0]*3
        longest = np.argmax(rng)
        tmp_resolution[longest] = resolution
        for i in range(n_dims):
            if i != longest:
                tmp_resolution[i] = int(resolution * rng[i] / np.max(rng))
          
        resolution = tmp_resolution
        
    
    coords = []
    for i in range(n_dims):
        coo = np.linspace(mins[i], maxs[i], resolution[i], endpoint=False)
        coords += [
            xr.DataArray(
                coo, coords = {name_coords[i]+"n":coo}
            )
        ]


    M = np.linalg.inv(a).T

    inv_coords = []
    for i in range(n_dims):
        inv_coo = 0
        for j in range(n_dims):
            inv_coo = inv_coo + M[i, j] * coords[j]
        inv_coords +=[inv_coo]
    
    return inv_coords

    

    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(
    #     xs = [-sum_a[0]/2, sum_a[0]/2],
    #     ys = [-sum_a[1]/2, sum_a[1]/2],
    #     zs = [-sum_a[2]/2, sum_a[2]/2],
    #     )
    # ax.set_xlim(-extrm[0]/2, extrm[0]/2)
    # ax.set_ylim(-extrm[1]/2, extrm[1]/2)
    # ax.set_zlim(-extrm[2]/2, extrm[2]/2)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()

# if __name__ == "__main__":
#     # from bloch_schrodinger.potential import create_parameter

#     dr = create_parameter("dr", [0, 0.1])

#     foo = Potential(
#         [[10, 0, 0], [0, 5, 0], [0, 0, 8]], resolution=(100, 50, 70), endpoint=True
#     )

#     dimp = create_parameter("dimp", np.linspace(0, 1, 50))
#     foo.set(foo.coords[0] ** 4)
#     foo.add((100 - 20 * foo.coords[0] ** 2) * dimp)

#     create_cart_grid(foo.V)
