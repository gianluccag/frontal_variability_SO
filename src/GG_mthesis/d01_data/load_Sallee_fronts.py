import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

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

def load_ACC_fronts(path, date, front_name):
    '''
    Parameters
    ----------
    path : string
        The path to each front.
    date: string
        The date in the following format: YYYY-MM-DD.
    front_name: string
        One of the following: PF, SAF_N, SAF. Has to go in pair with the corresponding path.

    Returns
    ----------
    latitude_front_name : numpyp array
        The latitude of each frontal position for the specified date.
    longitude_front_name : numpy array
        The longitude of each frontal position for the specified date. 
    '''

    from netCDF4 import Dataset
    latitude = Dataset(path).variables[front_name] #extract the latitude from the dataset
    longitude = Dataset(path).variables['longitude'] #extract the longitude from the dataset
    time = Dataset(path).variables['time'] #extract the time from the dataset

    days = (np.datetime64(date) - np.datetime64('1950-01-01')).astype(int) #calculate the numbers of days until your required date (because of time format)
    closest = closest_value(time, days) #closest day in the dataset to the date
    index = np.where(np.array(time) == int(closest)) #index of the day in the dataset

    lat_name = 'latitude_' + front_name #naming of the frontal latitude
    lon_name = 'longitude_' + front_name #naming of the frontal longitude

    globals()[lat_name] = latitude[index][0] #final selection (and globalisation) of the latitude.
    globals()[lon_name] = longitude[index][0] #final selection (and globalisation) of the longitude.


def plot_ACC_fronts(root_path, date='2017-12-05', central_longitude=-30, savepath=None):
    '''
    Parameters
    ----------
    root_path : string
        the path until the SALEE2008 folder.
    date: string
        The date in the following format: YYYY-MM-DD.
    central_longitude : float, int
        The central longitude for the map (-180,180 format).
    savepath : string
        Path to save the figure to, optional.

    Returns
    ----------
    Map of the SO with the ACC fronts for the given date.

    '''
    load_ACC_fronts(root_path + '/SALLEE2008_SO_FRONTS/CTOH_PolarFront_weekly_1993_2018.nc', date, 'PF')
    load_ACC_fronts(root_path + '/SALLEE2008_SO_FRONTS/CTOH_NorthernSubantarcticFront_weekly_1993_2018.nc', date, 'SAF_N')
    load_ACC_fronts(root_path + '/SALLEE2008_SO_FRONTS/CTOH_SubantarcticFront_weekly_1993_2018.nc', date, 'SAF')

    fig = plt.figure(figsize=(20,10))
    ax = plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=central_longitude))

    PF = ax.plot(longitude_PF, latitude_PF, transform=ccrs.PlateCarree(), label='PF', color='black')
    SAF_N = ax.plot(longitude_SAF_N, latitude_SAF_N,transform=ccrs.PlateCarree(), label='SAF_N', color='darkslategray')
    SAF = ax.plot(longitude_SAF, latitude_SAF,transform=ccrs.PlateCarree(), label='SAF', color='gray')

    # ds_ssh_SO.adt.sel(time='2017-11-09').plot.contourf(
    #     transform=ccrs.PlateCarree(),
    #     levels=30,
    #     cmap='Spectral_r',
    #     ax=ax)

    ax.coastlines()
    ax.gridlines(draw_labels=True,
        linewidth=2,
        color='gray',
        alpha=0.5,
        linestyle='--')

    plt.legend()

    if savepath != None:
        plt.savefig(savepath)

    plt.show()