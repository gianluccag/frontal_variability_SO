from fileinput import filename
import numpy as np
import pandas as pd
from datetime import datetime

class SOCAT():
    
    def read_dataset(filepath='data/01_raw/dataset.csv'):
        df = pd.read_csv(filepath)

        df = df.drop(['Unnamed: 0', 'version', 'GVCO2 [umol/mol]', 'xCO2water_equ_dry [umol/mol]', 'xCO2water_SST_dry [umol/mol]',
                'pCO2water_equ_wet [uatm]', 'pCO2water_SST_wet [uatm]', 'fCO2water_equ_wet [uatm]', 'fCO2water_SST_wet [uatm]', 'fCO2rec [uatm]',
                'fCO2rec_src', 'fCO2rec_flag', 'NCEP_SLP [hPa]', 'Pequ [hPa]', 'PPPP [hPa]', 'Source_DOI'
                ], axis=1)

        # df['longitude'] = df['longitude [dec.deg.E]']
        # df['latitude'] = df['latitude [dec.deg.N]']
        df = df.rename({'SST [deg.C]':'T', 
                        'longitude [dec.deg.E]':'lon', 
                        'latitude [dec.deg.N]':'lat'
                        }, 
                       axis=1
                       )
        # df = df.drop(['longitude [dec.deg.E]', 'latitude [dec.deg.N]'], axis=1)
        
        from GG_mthesis.d00_utils.utils import convert_lon_center 
        df['lon'] = convert_lon_center(df['lon'], center='Atlantic')
        return df
    
    def _add_time_column(df):
        df['mm'][df['ss'] >= 60] = df['mm'][df['ss'] >= 60] + df['ss'][df['ss'] >= 60]/60
        df['ss'][df['ss'] >= 60] = df['ss'][df['ss'] >= 60] - 60
        df['hh'][df['mm'] >= 60] = df['hh'][df['mm'] >= 60] + df['mm'][df['mm'] >= 60]/60
        df['mm'][df['mm'] >= 60] = df['mm'][df['mm'] >= 60] - 60
        df['day'][df['hh'] >= 24] = df['day'][df['hh'] >= 24] + df['hh'][df['hh'] >= 24]/24
        df['hh'][df['hh'] >= 24] = df['hh'][df['hh'] >= 24] - 24

        df['datetime'] = [np.datetime64(datetime(
            int(df['yr'].values[i]), 
            int(df['mon'].values[i]), 
            int(df['day'].values[i]), 
            int(df['hh'].values[i]), 
            int(df['mm'].values[i]), 
            int(df['ss'].values[i])
            ))
                            for i in range(len(df))]

        df = df.drop(['yr', 'mon', 'day', 'hh', 'mm', 'ss'], axis=1)
        return df
    
    def select_initial_date(df, date_ini='2000-01-01T00:00'):
        date_ini = np.datetime64('2000-01-01T00:00')
        df = df.where(
            df['datetime'] >= date_ini).dropna(how='all')
        return df
    
    def select_region(df, 
                      lon_min=-67, lon_max=-55, 
                      lat_min=-63, lat_max=-55, 
                      depth=2500, 
                      date_ini=np.datetime64('2000-01-01T00:00')):
        
        df = df.where(
            (df['lon'] > lon_min) & 
            (df['lon'] < lon_max) & 
            (df['lat'] > lat_min) & 
            (df['lat'] < lat_max)
            )
        
        df = df.where(df['ETOPO2_depth [m]'] >= depth).dropna(how='all')
        df = df.reset_index(drop=True)
        
        df = SOCAT._add_time_column(df)
        df = SOCAT.select_initial_date(df, date_ini=date_ini)

        return df
    
    def SA_CT_rho(df):
        import gsw
        df['SA'] = gsw.SA_from_SP(df['sal'].values, 0, df['lon'].values, df['lat'].values)
        df['CT'] = gsw.CT_from_t(df['SA'].values, df['T'].values,0)
        df['rho'] = gsw.rho(df['SA'].values, df['CT'].values, 0)
        return df
    
    def save_cruises_to_disc(df, path='data/02_intermediate/SOCAT/'):
        cruise_id = df['Expocode'].unique()
        for i in range(len(cruise_id)):
            df_aux = df[df['Expocode'] == cruise_id[i]].set_index('datetime')
            if df_aux.empty == False:
                path = path
                filename = path + str(cruise_id[i]) + '.csv'
                df_aux.to_csv(filename)
            else:
                continue

    def select_transect(df, path='data/03_processed/SOCAT/'):
        from GG_mthesis.d00_utils import geo
        bearing = [geo.calculateBearing(df['lat'][i],df['lon'][i],df['lat'][i+1],df['lon'][i+1]) for i in range(len(df['lon'])-1)]
        df = df[:-1]
        df['bearing'] = bearing
        
        df_south = df.where(df['bearing'] > 135).where(df['bearing'] < 225)
        df_south = df_south[df_south['bearing'].notna()]

        df_north = df[~df['bearing'].between(45,315)]
        df_north = df_north[df_north['bearing'].notna()]
        
        if not df_south.empty:
            filename_south = path + str(df_south['Expocode'][0]) + '_south.csv'
            df_south.to_csv(filename_south)

        if not df_north.empty:
            filename_north = path + str(df_north['Expocode'][0]) + '_north.csv'
            df_north.to_csv(filename_north)


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
        # time_stamp_arr = np.array([df['datetime'].iloc[i].to_pydatetime().timestamp() for i in range(len(df['datetime']))])
        # time = griddata(df['distance_cum'].values, time_stamp_arr, distance_grid, method='linear')
        # time = [np.datetime64(datetime.fromtimestamp(time[i])) for i in range(len(time))]
        df_aux = pd.DataFrame(rho, columns=['rho'], index=distance_grid)
        df_aux['lon'] = lon
        df_aux['lat'] = lat
        # df_aux['time'] = time
        return df_aux
    
    