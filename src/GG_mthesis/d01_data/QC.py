import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc

import nc_time_axis
import cftime
from datetime import datetime

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import traceback
from matplotlib import colors

def count_flags(variable):
    """Counts the ocurrence of each flag returning a dictionary with each flag description and the occurence.
    Args:
    variable(xr.DataArray, np.array, list, etc): the flag value of each datapoint in an array."""
    
    global QC_flags_id
    QC_flags_id = {48:'no_quality_control', 49:'good_value', 50:'probably_good_value', 51:'probably_bad_value',
                   52:'bad_value', 53:'changed_value', 54:'value_below_detection', 55:'value_in_excess',
                   56:'interpolated_value', 57:'missing_value',65:'value_phenomenon_uncertain'}
    
    global flags_name
    flags_name = []
    for name in QC_flags_id:
        flags_name.append(QC_flags_id[name])
    
    global flags_num
    flags_num = [48,49,50,51,52,53,54,55,56,57,65]
    
    count = {}
    for flag in flags_num:
        count[flag] = len(variable.loc[variable == flag])
        count[QC_flags_id[flag]] = count.pop(flag)
    return count


def cruise_map_QC(variable,longitude, latitude, projection='PlCar',
                  extent=False, savepath=False):

    """Visualize QC of a variable on a map.
    Args:
        longitude(np.array, pd.array, xr.DataArray): value array of longitude of each position.
        latitude(np.array, pd.array, xr.DataArray): value array of latitude of each position.
        proyection(str): desired proyection. Default = PlCar (PlateCarree). Other = TMerc (Transverse Mercator).
        extent(list): list of four floats indicating is this order [min_x, max_x, min_y, max_y].
        savepath(str or path-like): A path, or a python file-like object.
    """    
    
    count_flags(variable)
    
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

    cmap = colors.ListedColormap(['r','b','c','m','yellow','purple','orange', 'pink','black', 'gray','darkslategray'])
    ax.scatter(longitude, latitude, c=variable, cmap=cmap, alpha=1, s=0.05, transform=data_crs, 
               label=QC_flags_id)
    
    ax.stock_img()
    #ax.coastlines()
    ax.gridlines(draw_labels=True,
        linewidth=2,
        color='gray',
        alpha=0.5,
        linestyle='--')
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.LAKES)
    ax.add_feature(cartopy.feature.RIVERS)
    
    if savepath != False:
        plt.savefig(savepath, format='png',bbox_inches='tight')

    plt.show()