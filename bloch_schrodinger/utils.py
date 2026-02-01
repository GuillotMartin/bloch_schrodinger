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

