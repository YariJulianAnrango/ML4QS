import pandas as pd
import numpy as np
from tqdm import tqdm


class create_dataset:
    def __init__(self, weather_dir, delta_t):
        self.weather_dir = weather_dir
        self.delta_t = delta_t

    def fix_weather_data(self,rows_skipped=30):

        print('fixing weather data')
        weather_raw = pd.read_csv(self.weather_dir, dtype=str, skiprows=rows_skipped)

        # replace confusing names with better names derived from .txt file
        weather = weather_raw.rename(columns={'   HH': 'hour',
                                          '   DD': 'direction_wind',
                                          '   FH': 'windspeed_avg_hour',
                                          '   FF': 'windspeed_avg_10min',
                                          '   FX': 'max_wind_gust',
                                          '    T': 'temp_celsius',
                                          ' T10N': 'temp_min_6h',
                                          '   TD': 'temp_dewpoint',
                                          '   SQ': 'sunshine_duration',
                                          '    Q': 'glob_radiation',
                                          '   DR': 'precipitation_duration',
                                          '   RH': 'precipitation_amount_hourly',
                                          '    P': 'air_pressure',
                                          '   VV': 'horizontal_visibility',
                                          '    N': 'cloud_cover',
                                          '    U': 'relative_humidity',
                                          '   WW': 'weather_code',
                                          '   IX': 'indicator_present_weather_code',
                                          '    M': 'fog',
                                          '    R': 'rain',
                                          '    S': 'snow',
                                          '    O': 'thunder',
                                          '    Y': 'ice_formation'}, inplace=False)

    # remove whitespace from values
        for i in weather.columns:
            weather[i] = weather[i].str.strip()

        weather = weather.replace('', np.nan, regex=True)

    # change datatype
        for i in weather.columns:
            if i == 'precipitation_amount_hourly':
                weather[i] = weather[i].astype(float)
            else:
                weather[i] = weather[i].astype(float).astype("Int32")

    # fix datetimes
        weather['hour'] = pd.to_datetime(weather['hour'] - 1, format='%H', exact=False).dt.strftime(
            '%H:%M:%S')  # hour minus one because somehow 24:00:00 gets converted to 02:00:00, needs to be fixed
        weather['YYYYMMDD'] = pd.to_datetime(weather['YYYYMMDD'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        weather['datetime'] = weather[['YYYYMMDD', 'hour']].apply(lambda x: ' '.join(x.values.astype(str)), axis="columns")
        weather['datetime'] = pd.to_datetime(pd.to_datetime(weather['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S'))
        weather['start_weather'] = weather['datetime']
        weather['end_weather'] = pd.to_datetime(weather['start_weather']) + pd.Timedelta('1H')
        cols = list(weather)
        cols.insert(0, cols.pop(cols.index('end_weather')))
        cols.insert(0, cols.pop(cols.index('start_weather')))
        cols.pop(cols.index('hour'))
        cols.pop(cols.index('YYYYMMDD'))
        weather = weather.loc[:, cols]
        return weather

    def fix_steps_time_intervals(self, dir, start_date, end_date, dataframe_type:str):
        delta_timeformat = str(self.delta_t)+'min'
        print('fixing steps data')
        dat = pd.read_csv(dir)
        dat = dat[pd.to_datetime(dat['startDate']).dt.strftime('%Y-%m') == pd.to_datetime(start_date).strftime(
            '%Y-%m')].reset_index(drop=True)
        dat_start = pd.to_datetime(pd.to_datetime(dat['startDate']).dt.strftime('%Y-%m-%d %H:%M:%S'))
        dat_end = pd.to_datetime(pd.to_datetime(dat['endDate']).dt.strftime('%Y-%m-%d %H:%M:%S'))

        range_start = pd.to_datetime(pd.date_range(start=start_date,
                                               end=end_date,
                                               freq='S'))
        range_end = pd.to_datetime(pd.date_range(start=pd.to_datetime(start_date) + pd.Timedelta(seconds=1),
                                             end=pd.to_datetime(end_date) + pd.Timedelta(seconds=1),
                                             freq='S'))
        new_dat = pd.DataFrame({'start': range_start,
                              'end': range_end,
                              'value_'+dataframe_type: [0] * len(range_start)})

        for i in tqdm(range(len(dat_start))):
            mask = (new_dat['start'] >= dat_start[i]) & (new_dat['end'] <= dat_end[i])
            trues = len(mask[mask == True])
            if trues != 0:
                res = dat.loc[i, 'value'] / trues
                new_dat.loc[mask, 'value_'+dataframe_type] = res

        new_dat = new_dat.resample(delta_timeformat, on='end')['value_'+dataframe_type].sum().reset_index().rename(columns={'end': 'start'})
        new_dat['end'] = new_dat['start'] + pd.Timedelta(delta_timeformat)

        new_dat = new_dat[['start', 'end', 'value_'+dataframe_type]]

        return new_dat


    def fix_heartrate(self, dir, start_date, end_date, merged_df, key):
        print('fixing heartrate data')
        heart = pd.read_csv(dir)
        heart = heart[pd.to_datetime(heart['startDate']).dt.strftime('%Y-%m') >= pd.to_datetime(start_date).strftime(
            '%Y-%m')].reset_index(drop=True)
        merged_df['value_'+key] = 0

        heart['startDate'] = pd.to_datetime(pd.to_datetime(heart['startDate']).dt.strftime('%Y-%m-%d %H:%M:%S'))
        heart['endDate'] = pd.to_datetime(pd.to_datetime(heart['endDate']).dt.strftime('%Y-%m-%d %H:%M:%S'))

        for i in tqdm(range(len(merged_df))):
            mask = (pd.to_datetime(merged_df.loc[i,'start']) <= heart['startDate']) & (pd.to_datetime(merged_df.loc[i,'end']) >= heart['startDate'])
            trues = len(mask[mask==True])
            if trues != 0:
                res_sum = heart.loc[mask, 'value'].sum()
                res = res_sum/trues
                merged_df.loc[i,'value_'+key] = res

        return merged_df

    def fix_workout(self, dir, start_date, end_date, merged_df, key):
        print('fixing workout data')
        work = pd.read_csv(dir)
        work = work[pd.to_datetime(work['startDate']).dt.strftime('%Y-%m') >= pd.to_datetime(start_date).strftime(
            '%Y-%m')].reset_index(drop=True)
        merged_df['value_' + key] = 'no workout'

        work['startDate'] = pd.to_datetime(pd.to_datetime(work['startDate']).dt.strftime('%Y-%m-%d %H:%M:%S'))
        work['endDate'] = pd.to_datetime(pd.to_datetime(work['endDate']).dt.strftime('%Y-%m-%d %H:%M:%S'))

        for i in tqdm(range(len(work))):
            mask1 = (pd.to_datetime(merged_df['start']) <= work.loc[i,'startDate']) & (
                        pd.to_datetime(merged_df['end']) >= work.loc[i,'startDate'])
            mask2 = (pd.to_datetime(merged_df['start']) >= work.loc[i,'startDate']) & (
                        pd.to_datetime(merged_df['end']) <= work.loc[i,'endDate'])
            mask3 = (pd.to_datetime(merged_df['start']) <= work.loc[i,'endDate']) & (
                        pd.to_datetime(merged_df['end']) >= work.loc[i,'endDate'])

            mask = pd.Series(map(any, zip(*[mask1, mask2, mask3])))

            trues = len(mask[mask == True])
            if trues > 0:
                workout = work.loc[i, 'workoutActivityType']
                merged_df.loc[mask, 'value_' + key] = workout

        return merged_df

    def merge_data_weather(self, dir_steps: str, dir_heart: str, dir_workout: str, start_date, end_date,  steps_column_datetime='start'):
        steps = self.fix_steps_time_intervals(dir_steps, start_date, end_date, 'steps')
        weather = self.fix_weather_data()

        print('merging weather and steps data')

        # decide on column to merge by, change steps_column_datetime to choose another datetime column
        weather.rename(columns={'datetime': steps_column_datetime}, inplace=True)

        steps[steps_column_datetime] = pd.to_datetime(
            pd.to_datetime(steps[steps_column_datetime]).dt.strftime('%Y-%m-%d %H:%M:%S'))
        steps.sort_values(steps_column_datetime, inplace=True)
        weather.sort_values(steps_column_datetime, inplace=True)

        merge = pd.merge_asof(steps, weather, on=steps_column_datetime, tolerance=pd.Timedelta("60m"))


        merge_heart = self.fix_heartrate(dir_heart, start_date, end_date, merge, 'heart')
        merge_workout = self.fix_workout(dir_workout, start_date, end_date, merge_heart, 'workout')

        self.data_frame = merge_workout

        self.construct_final_df(['start','end','value_steps','value_heart','value_workout','temp_celsius','rain'])
    def construct_final_df(self, cols):
        self.data_frame = self.data_frame[cols]



data = create_dataset('C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\weather_2022_2023.txt', delta_t=10)
data.merge_data_weather(dir_steps='C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\StepCount.csv',
                        dir_heart='C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\HeartRate.csv',
                        dir_workout='C:\\Users\\irene\\OneDrive\\Bureaublad\\ML\\ML4QS\\data_used\\Workout.csv',
                        start_date='2022-01-01 00:00:00',
                        end_date='2023-06-06 00:00:00')

data.data_frame.to_csv('New_weather_steps.csv')




