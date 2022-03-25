def front_loc(ds_latitude, ds_ssh_grad, grad_threshold, n_dim = 3, type = 'map', time_dim='time', lat_dim='latitude', lon_dim='longitude'):
    '''
    Provides an array with a value of 1 at points (latitude and time) where the gradient is larger than a threshold value. Every
    other value is converted into a NaN. This facilitas the creation of contour maps (e.g. Hovmoller).
    Adapted from Sokolov, S. and S. R. Rintoul (2007). "Multiple Jets of the Antarctic Circumpolar Current South of Australia*." 
    Journal of Physical Oceanography 37(5): 1394-1412.
    
    Parameters
    ----------
    ds_latitude :xarray array
        The latitude variable in the xarray dataset. 
    ds_ssh_grad : xarray array
        The ssh gradient variable in the xarray dataset in m/km.
    grad_threshold : float
        SSH gradient threshold in m/100km.
    type : str
        Kind of figure the data is going to be for. Either 'map' or 'hovmoller', default: 'map'.
    time_dim : str
        Name of the time dimension in the dataset, default: 'time'.
    lat_dim : str
        Name of the latitude dimension in the dataset, default: 'latitude'.
    lon_dim : str
        Name of the longitude dimension in the dataset, default: 'longitude'.

    Returns
    ----------
    front_loc : array of 1 and NaNs
        For each date it provides the latitude at which the SSH gradient surpasses the given threshold.
    '''
    import numpy as np
    import xarray as xr
    import metpy

    #This selects the latitudes for each date where the gradient is larger than the threshold. The rest is converted in NaNs.
    front_loc = (ds_latitude.where(np.abs((ds_ssh_grad*100).metpy.dequantify()) >= grad_threshold)) 
    #This converts the new variable in booleans. True if it is a number, 0 if it is a NaN.
    front_loc = front_loc.fillna(0).astype(bool)
    #Reorganizes the dimensions.
    if type == 'map':
        if n_dim == 3:
            front_loc = front_loc.transpose(time_dim,lat_dim, lon_dim)
        elif n_dim == 2:
            front_loc = front_loc.transpose(lat_dim, lon_dim)
        else:
            raise ValueError('Either 2 dimensions (time and latitude) or 3 dimensions (time, latitude and longitude) are accepted.')
    elif type == 'hovmoller':
        front_loc = front_loc.transpose(time_dim,lat_dim)
    #This converts each value into 1 (greater than threshold) or NaNs. 
    front_loc = front_loc.where(front_loc == True)

    return front_loc



