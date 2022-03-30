import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy 
import cartopy.crs as ccrs

class rmap():    
    def load(filepath='data/01_raw/ETOPO1_Bed_g_gmt4.grd', engine='netcdf4'):
        etopo1 = xr.open_dataset(filepath, engine=engine)
        return etopo1
    
    def select_region(dataset, lon_min=-75, lon_max=-52, lat_min=-65, lat_max=-52):
        bath = dataset.sel(x=slice(lon_min,lon_max), y=slice(lat_min,lat_max))
        return bath
    
    def load_fronts(filepath='data/03_processed/geo_fronts/park/62985.nc'):
        ds_ACC_fronts = xr.load_dataset(filepath)
        return ds_ACC_fronts
    
    def sel_front_region(dataset, lon_min=-75, lon_max=-52, lat_min=-65, lat_max=-52):
        data_lat = [
            dataset['LatNB'].values,
            dataset['LatSAF'].values,
            dataset['LatPF'].values,
            dataset['LatSACCF'].values,
            dataset['LatSB'].values,
        ]

        data_lon = [
            dataset['LonNB'].values,
            dataset['LonSAF'].values,
            dataset['LonPF'].values,
            dataset['LonSACCF'].values,
            dataset['LonSB'].values,
        ]

        index_front = ['NB', 'SAF', 'PF', 'SACCF', 'SB']

        df_fronts = pd.DataFrame(columns=['longitude','latitude','front'])
        for i in range(5):
            df_aux = pd.DataFrame({'longitude':data_lon[i], 'latitude':data_lat[i],'front':index_front[i]})
            df_fronts = df_fronts.append(df_aux)
        df_fronts = df_fronts.reset_index(drop=True)

        df_fronts_sel = df_fronts.where(
            (df_fronts['longitude'] > lon_min) & 
            (df_fronts['longitude'] < lon_max) & 
            (df_fronts['latitude'] > lat_min) & 
            (df_fronts['latitude'] < lat_max)
            ).dropna().reset_index()
        df_front_sel = df_fronts_sel.set_index(['front',df_fronts_sel.index]).drop('index',1)
        return df_front_sel
        
    def plot_region(filepath_bath='data/01_raw/ETOPO1_Bed_g_gmt4.grd', projection=ccrs.PlateCarree(), 
                    fronts=True, filepath_fronts='data/03_processed/geo_fronts/park/62985.nc',
                    lon_min=-75, lon_max=-52, lat_min=-65, lat_max=-52):
        bath = rmap.load(filepath_bath)
        bath = rmap.select_region(bath, lon_min, lon_max, lat_min, lat_max)
        import cmocean.cm as cm
        from matplotlib.cm import ScalarMappable
        fig = plt.figure(figsize=(14,12))
        ax = plt.axes(projection=projection)
        x = bath.x
        y = bath.y
        z = bath.z
        vmin = z.min()
        vmax = z.max()
        topo_f = ax.contourf(x, y, z, 
                cmap = cm.deep_r,
                    vmin = vmin,
                    vmax = 0,
                #    norm=MidpointNormalize(
                #        midpoint=-1000,
                #        vmin=vmin,lines
                #        vmax=0
                #     ),
                levels=20
                )
        #plt.colorbar(ScalarMappable(norm=topo_f.norm, cmap=topo_f.cmap), pad=0.02)
        
        ax.coastlines(zorder=100)
        ax.add_feature(cartopy.feature.LAND, color='green', zorder=50)
        gl = ax.gridlines(draw_labels=True, linestyle='-.', color='gray')
        gl.xlabels_top = False
        gl.ylabels_right = False

        if fronts == True:
            fronts=rmap.load_fronts(filepath_fronts)
            fronts=rmap.sel_front_region(fronts, lon_min, lon_max, lat_min, lat_max)
            s=4
            SAF = ax.plot(fronts.loc['SAF']['longitude'], fronts.loc['SAF']['latitude'],transform=ccrs.PlateCarree(), label='SAF_Park', color='purple')#, s=s)
            PF = ax.plot(fronts.loc['PF']['longitude'], fronts.loc['PF']['latitude'], transform=ccrs.PlateCarree(), label='PF_Park', color='black')#, s=s)
            SACCF = ax.plot(fronts.loc['SACCF']['longitude'], fronts.loc['SACCF']['latitude'],transform=ccrs.PlateCarree(), label='SACCF_Park', color='yellow')#, s=s)
            NB = ax.plot(fronts.loc['NB']['longitude'], fronts.loc['NB']['latitude'],transform=ccrs.PlateCarree(), label='NB_Park', color='red')#, s=s)
            SB = ax.plot(fronts.loc['SB']['longitude'], fronts.loc['SB']['latitude'],transform=ccrs.PlateCarree(), label='SB_Park', color='red')#, s=s)
        return ax

def sort_transect(df):
    """
    Sorts the data so it is aligned from North to South. This is important for calculating the gradient, so it is calculated in the same direction.
    
    Args:
        df (pd.dataframe): the dataframe with the preprocessed TSG data, latitude should be given as lat and the index should be time.

    Returns:
        pd.dataframe: the same dataframe sorted from North to South.
    """
    if df['lat'][df['lat'].first_valid_index()] < df['lat'][df['lat'].last_valid_index()]:
        df = df.sort_index(ascending=False)
    return df

def downsample(df, gridsize):
    """
    Grids the data to a equally distance grid.

    Args:
        df (pd.dataframe): pandas dataframe with the North-South sorted TSG transect.
        gridsize (float): the desired distance it should be gridded to.

    Returns:
        pd.dataframe: gridded data.
    """
    from scipy.interpolate import griddata
    from datetime import datetime
    #create the distance grid
    distance_grid = np.arange(0, float(np.max(df.distance_cum)), gridsize)
    
    #we need to create a mask so the nans stay nans after gridding and are not interpolated
    df_mask = df.notna()[['rho', 'distance_cum']]
    df_mask['distance_cum'] = df['distance_cum']
    grid_mask = griddata(df_mask['distance_cum'].values, df_mask['rho'].values, distance_grid, method='linear')

    rho = griddata(df['distance_cum'].values, df['rho'].values, distance_grid, method='linear')*grid_mask
    lon = griddata(df['distance_cum'].values, df['lon'].values, distance_grid, method='linear')
    lat = griddata(df['distance_cum'].values, df['lat'].values, distance_grid, method='linear')
    time_stamp_arr = np.array([df['time'].iloc[i].to_pydatetime().timestamp() for i in range(len(df['time']))])
    time = griddata(df['distance_cum'].values, time_stamp_arr, distance_grid, method='linear')
    time = [np.datetime64(datetime.fromtimestamp(time[i])) for i in range(len(time))]
    df_aux = pd.DataFrame(rho, columns=['rho'], index=distance_grid)
    df_aux['lon'] = lon
    df_aux['lat'] = lat
    df_aux['time'] = time
    return df_aux

def calculate_bx(df):
    """
    Calculates the horizontal buoyancy gradient for a transect over 1km. Can only be used for already distance gridded data. The density should be named rho.

    Args:
        df (pd.dataframe): dataframe with the transect data, should contain rho.
        
    Returns:
        A new variable in the dataframe df named bx.
    """
    g=9.81
    rho_0 = 1025
    df['bx'] = (-g/rho_0)*df['rho'].diff()/1000

def truncate(number, digits):
    import math as m
    stepper = 10 ** digits
    return m.trunc(number*stepper)/stepper

def grid_limits(df):
    if df['lat'].max() - truncate(df['lat'].max(), 2) < -0.005:
        lat_max_north = truncate(df['lat'].max(), 2)
    else:
        lat_max_north = truncate(df['lat'].max(), 2) - 0.005
        
    if df['lat'].min() - truncate(df['lat'].min(), 2) < -0.005:
        lat_max_south = truncate(df['lat'].min(), 2)
    else:
        lat_max_south = truncate(df['lat'].min(),2) + 0.005
        
    return lat_max_north, lat_max_south

def downsample_lat(df, gridsize, bx='bx', n_lim=-55.0, s_lim=-60.5, method='linear'):
    """
    Interpolates to a constant latitude grid. 

    Args:
        df (pd.dataframe): TSG data including latitude (lat), longitude (lon), density (rho) and the horizontal buoyancy gradient (bx).
        gridsize (float): latitude size in degrees.
        lat_max(float): most northern latitude.
        lat_min(float): most southern latitude.

    Returns:
        pd.dataframe: latitude gridded data. 
    """
    import math as m
    from scipy.interpolate import griddata
    from datetime import datetime
    from GG_mthesis.d01_data.processing import truncate, grid_limits
    #create the distance grid
    lat_max, lat_min = grid_limits(df)
    lat_grid = np.arange(int(lat_max*1000), int(lat_min*1000), int(gridsize*1000))/1000
    
    #we need to create a mask so the nans stay nans after gridding and are not interpolated
    df_mask = df.notna()[['rho', bx, 'lat']]
    df_mask['lat'] = df['lat']
    grid_mask = griddata(df_mask['lat'].values, df_mask['rho'].values, lat_grid, method='nearest')
    
    x = df['lat'].values
    xp = lat_grid
    rho_aux = griddata(x, df['rho'].values, xp, method=method)*grid_mask
    bx_aux = griddata(x, df[bx].values, xp, method=method)*grid_mask
    lon_aux = griddata(x, df['lon'].values, xp, method=method)
    
    time_stamp_arr = np.array([df['time'].iloc[i].to_pydatetime().timestamp() for i in range(len(df['time']))])
    time = griddata(df['lat'].values, time_stamp_arr, lat_grid, method=method)
    time = [datetime.fromtimestamp(time[i]) for i in range(len(time))]
    
    df_aux = pd.DataFrame(rho_aux, columns=['rho'], index=lat_grid)
    df_aux['bx'] = pd.DataFrame(bx_aux, columns=['bx'], index=lat_grid)
    df_aux['lon'] = pd.DataFrame(lon_aux, columns=['lon'], index=lat_grid)
    df_aux['lat'] = lat_grid
    df_aux['time'] = time

    aux_north = pd.DataFrame(index=np.arange(int(n_lim*1000), int(lat_max*1000), int(gridsize*1000))/1000) 
    #this is because of floating point arithmetic, which creates small imprecisions when repeatedly adding decimal numbers (what arange does). To circumvent this issue use
    #integers when possible
    aux_north[['rho', 'bx', 'lon', 'lat']] = np.nan, np.nan, np.nan, aux_north.index

    aux_south = pd.DataFrame(index=np.arange(int(lat_min*1000), int((-60.5+gridsize)*1000), int(gridsize*1000))/1000)
    aux_south[['rho', 'bx', 'lon', 'lat', 'time']] = np.nan, np.nan, np.nan, aux_south.index, np.nan    
    df_aux = pd.concat([aux_north, df_aux, aux_south])

    return df_aux

def method_1(df):
    g=9.8
    rho_0 = 1025
    df_500 = downsample(df,500)
    df_500['bx_500'] = (-g/rho_0)*df_500['rho'].diff()/1000
    df_500['bx_1000'] = (-g/rho_0)*df_500['bx_500'].rolling(window=2).mean()
    df_500['bx_2000'] = (-g/rho_0)*df_500['bx_500'].rolling(window=4).mean()
    df_500['bx_5000'] = (-g/rho_0)*df_500['bx_500'].rolling(window=10).mean()
    df_500['bx_10000'] = (-g/rho_0)*df_500['bx_500'].rolling(window=20).mean()
    return df_500

def method_3(df):
    g=9.8
    rho_0 = 1025
    df_500 =  downsample(df, 500)
    df_500['rho_1000'] = df_500['rho'].rolling(2).mean()
    df_500['rho_2000'] = df_500['rho'].rolling(4).mean()
    df_500['rho_5000'] = df_500['rho'].rolling(10).mean()
    df_500['rho_10000'] = df_500['rho'].rolling(20).mean()
    df_500['bx_500'] = (-g/rho_0)*df_500['rho'].diff()/1000
    df_500['bx_1000'] = (-g/rho_0)*df_500['rho_1000'].diff()/1000
    df_500['bx_2000'] = (-g/rho_0)*df_500['rho_2000'].diff()/1000
    df_500['bx_5000'] = (-g/rho_0)*df_500['rho_5000'].diff()/1000
    df_500['bx_10000'] = (-g/rho_0)*df_500['rho_10000'].diff()/1000
    return df_500