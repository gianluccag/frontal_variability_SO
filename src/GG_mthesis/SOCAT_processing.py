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