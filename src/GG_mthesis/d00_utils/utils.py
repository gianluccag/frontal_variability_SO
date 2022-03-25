def closest_value(array, value):
    '''
    Provides the closest value in an array to a given value.
    
    Parameters
    ----------
    array : array_like
        Array to search in.
    value : float
        Value to find in the array.

    Returns
    ----------
    closest_val
        The closest value in the array to the given value as a float.
    '''
    import numpy as np

    closest_val = array[(np.abs(np.array(array) - value).argmin())]
    return float(closest_val)

def convert_lon_center(values, center='Atlantic'):
    '''
    Converts the longitude to a format centered either in the Atlantic (-180° to 180°) or in the Pacific (0° to 360°).

    Parameters
    ----------
    values : int or float or array_like
        The longitude value(s) to convert.
    center : ObjectSelector, default: 'Atlantic'
        Wheter the longitude values should be referenced centered to the Atlantic or to the Pacific.

    Returns
    ----------
    longitude
        float or array of floats of the converted longitude value(s).
    '''
    if center=='Atlantic':
        longitude = ((values + 180) % 360) - 180
    else:
        longitude = values % 360

    return longitude

def np_dropna(array):
    import numpy as np
    '''
    Drops nan values from a numpy array and returns the same array without the nan values.
    '''
    nan_array = np.isnan(array)
    non_nan_array = ~ nan_array
    array_new = array[non_nan_array]
    return array_new