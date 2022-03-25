import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

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
        
    ax.plot(np.array(longitude), np.array(latitude), c='red', alpha=1, transform=data_crs, linestyle='--')
    ax.stock_img()
    #ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.LAKES)
    ax.add_feature(cartopy.feature.RIVERS)

    if savepath != False:
        plt.savefig(savepath, format='png',bbox_inches='tight')

    plt.show()

### Make SO plot boundary a circle
def plot_circle_boundary(ax) -> None:
    import matplotlib.path as mpath
    """
    Make SO plot boundary a circle.
    Compute a circle in axes coordinates, which we can use as a boundary for the map.
    We can pan/zoom as much as we like - the boundary will be permanently circular.
    """
    theta  = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5  ## could use 0.45 here, as Simon Thomas did
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform = ax.transAxes)