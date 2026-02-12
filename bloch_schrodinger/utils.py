import numpy as np
import xarray as xr
from typing import Union
from ipywidgets import FloatSlider, IntSlider



def islinspace(arr:xr.DataArray)->tuple[bool, float]:
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

def create_sliders(arr:xr.DataArray, list_dims:list[str], start:str = 'left')->dict[Union[IntSlider,FloatSlider]]:
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
        val = coord.min() if start == 'left' else (coord.max()-coord.min())/2
        islin, step = islinspace(coord)
        if np.issubdtype(coord.dtype, np.floating):
            sliders_ax[dim] = FloatSlider(
                min=float(coord.min()),
                max=float(coord.max()),
                step = float((coord.max() - coord.min()) / max(100, len(coord))) if not islin else step,
                value=float(val),
                description=dim
            )
        else:
            sliders_ax[dim] = IntSlider(
                min=coord.min(), 
                max=coord.max(), 
                step=1 if not islin else step,
                value=val, 
                description=dim
            )
    return sliders_ax

def create_sliders_from_dims(coordinates:dict[xr.DataArray], start:str = 'left')->dict[Union[IntSlider,FloatSlider]]:
    """Create a dictionnary of sliders from a dictionnary of dimensions.

    Args:
        coordinates (dict[xr.DataArray]): The dimensions for which to create the sliders.
        start (str): The default position of sliders, either 'left', 'right' or 'mid'. default to left.

    Returns:
        dict[FloatSlider]: _description_
    """

    sliders_ax = {}
    for dim, coord in coordinates.items():
        val = coord.min().item() if start == 'left' else (coord.max().item()-coord.min().item())/2
        islin, step = islinspace(coord)
        if np.issubdtype(coord.dtype, np.floating):
            sliders_ax[dim] = FloatSlider(
                min=float(coord.min()),
                max=float(coord.max()),
                step = float((coord.max() - coord.min()) / max(100, len(coord))) if not islin else step,
                value=float(val),
                description=dim
            )
        else:
            sliders_ax[dim] = IntSlider(
                min=coord.min(), 
                max=coord.max(), 
                step=1 if not islin else step,
                value=val, 
                description=dim
            )
    return sliders_ax



def coarsen(arr:xr.DataArray, factor: tuple[int, int]) -> xr.DataArray:
    """Return a coarsened version of the array, with reduced resolution along a1 and a2.

    Args:
        arr (xr.DataArray): The array to coarsen
        factor (tuple[int, int]): The coarsening factor. The initial resolution must be a multiple of the factor.

    Returns:
        Potential: Coarsened array
    """
    carr = arr.coarsen(a1=factor[0], a2=factor[1]).mean()
    return carr

def tile(arr:xr.DataArray, unitvecs:list[list[float, float]] , bounds1: tuple[int, int], bounds2: tuple[int, int]) -> xr.DataArray:
    """Return a copy of the array extended over multiple unit cells by tiling the original pattern.

        Args:
            arr (xr.DataArray): The array to tile
            unitvecs (list[list[float, float]]): The unit vectors to use for tiling.
            bounds1 (tuple[int, int]): Numbers of cells along a1
            bounds2 (tuple[int, int]): Number of cells along a2

        Returns:
            xr.DataArray
        """

    coords = {dim:arr.coords[dim] for dim in arr.dims if dim not in ['a1', 'a2']}
    
    reps1 = (bounds1[1]-bounds1[0]) 
    reps2 = (bounds2[1]-bounds2[0]) 
    na1, na2 = arr.sizes['a1'], arr.sizes['a2']
    na1_tot = na1 * reps1
    na2_tot = na2 * reps2
    
    coords.update(
        {'a1':np.linspace(bounds1[0]-1/2, bounds1[1]-1/2, na1_tot, endpoint=False),
            'a2':np.linspace(bounds2[0]-1/2, bounds2[1]-1/2, na2_tot, endpoint=False)}
    )
    
    shape = tuple(
        [coord.shape[0] for coord in coords.values()]
    )
    
    new_arr = xr.DataArray(
        np.zeros(shape, dtype = arr.dtype),
        coords=coords
    )
    
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
    
    new_arr = new_arr.assign_coords(
        {"x": tot_x, "y": tot_y}
    )

    # Loop da loop (sorry)
    for ia1 in range(reps1):
        for ia2 in range(reps2):
            lcR = {
                'a1':slice(ia1*na1, (ia1+1)*na1),
                'a2':slice(ia2*na2, (ia2+1)*na2)
            }
                            
            new_arr[lcR] += arr.transpose(..., 'a1', 'a2').data
    
    return new_arr
