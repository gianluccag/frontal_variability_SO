import numpy as np
import xarray as xr
import nc_time_axis
import cftime
from datetime import datetime
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

def cruise_map(longitude, latitude, projection = 'PlCar', extent=False, savepath=False):
    """Visualize the sampling positions as a line transects.
    Args:
        longitude(np.array, pd.array, xr.DataArray): value array of longitude of each position.
        latitude(np.array, pd.array, xr.DataArray): value array of latitude of each position.
        proyection(str): desired proyection. Default = PlCar (PlateCarree). Other = TMerc (Transverse Mercator).
        extent(list): list of four floats indicating the extent in this order [min_x, max_x, min_y, max_y]. 
    """    
    import traceback
    
    data_crs = ccrs.PlateCarree()
    
    try:
        central_lon = np.median(longitude); central_lat = np.median(latitude)
        
    except:
        traceback.print_exc()
        
    try:
        if projection == 'PlCar':
            projection = ccrs.PlateCarree(central_longitude=central_lon)
        elif projection == 'TMerc':
            projection = ccrs.TransverseMercator(central_longitude=central_lon, central_latitude=central_lat )
    except:
        traceback.print_exc()

    fig = plt.figure(figsize=(20, 10))

    ax = plt.axes(projection=projection)
     #This sets up GeoAxes instance exposing a variety of other map related methods
        
    if extent != False:
        x1, x2, y1, y2 = extent
        ax.set_extent(extent)
        
    ax.scatter(np.array(longitude), np.array(latitude), c='red', alpha=1, s=0.05, transform=data_crs)
    ax.stock_img()
    #ax.coastlines()
    ax.gridlines(draw_labels=True,
        linewidth=2,
        color='gray',
        alpha=0.5,
        linestyle='--')
    ax.gridlines(zorder=0)
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.LAKES)
    ax.add_feature(cartopy.feature.RIVERS)

    if savepath != False:
        plt.savefig(savepath, format='png',bbox_inches='tight')

    plt.show()

def select_transect(ds, min_lon, max_lon, min_lat, max_lat, min_time='1990-01-01', max_time='2100-12-31', 
    greater_equal_min_time=True, greater_equal_max_time=True):
    """Selects the data by latitude, longitude and time.
    Args:
        ds(xr.dataset): the xarray dataset.
        min_lon(float): minimum longitude from -180° to 180°.
        max_lon(float): maximum longitude from -180° to 180°.
        min_lat(float): minimum latitude from -90° to 90°.
        max_lat(float): maximum latitude from -90° to 90°.
        min_time(str): starting time in datetime64. Format:'year-month-day'. Default is 1990-01-01.
        max_time(str): ending time in datetime64. Format:'year-month-day'. Defult is 2100-12-31.
        """
    min_time = np.array([min_time], dtype='datetime64')
    max_time = np.array([max_time], dtype='datetime64')
    
    if greater_equal_min_time == True and greater_equal_max_time == True:
        ds_transect = ds.where((ds.lon >= min_lon) & 
                            (ds.lon <= max_lon) & 
                            (ds.lat >= min_lat) & 
                            (ds.lat <= max_lat) &
                            (ds.time >= min_time) &
                            (ds.time <= max_time),
                            drop=True)
    elif greater_equal_min_time == False and greater_equal_max_time == False: 
        ds_transect = ds.where((ds.lon >= min_lon) & 
                        (ds.lon <= max_lon) & 
                        (ds.lat >= min_lat) & 
                        (ds.lat <= max_lat) &
                        (ds.time > min_time) &
                        (ds.time < max_time),
                        drop=True)

    elif greater_equal_min_time == True and greater_equal_max_time == False: 
        ds_transect = ds.where((ds.lon >= min_lon) & 
                        (ds.lon <= max_lon) & 
                        (ds.lat >= min_lat) & 
                        (ds.lat <= max_lat) &
                        (ds.time >= min_time) &
                        (ds.time < max_time),
                        drop=True)

    elif greater_equal_min_time == False and greater_equal_max_time == True: 
        ds_transect = ds.where((ds.lon >= min_lon) & 
                        (ds.lon <= max_lon) & 
                        (ds.lat >= min_lat) & 
                        (ds.lat <= max_lat) &
                        (ds.time > min_time) &
                        (ds.time <= max_time),
                        drop=True)                        
    return ds_transect