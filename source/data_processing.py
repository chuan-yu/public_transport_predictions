import numpy as np
import pandas as pd
import xarray as xr
import os

# Load Data
file_path="../data/RIDE_DATA_OCT_2016-Bus2-wo-CAN.csv"
df = pd.DataFrame.from_csv(file_path, index_col=None)
df = df[df["Direction"]==0]
df['Ride_start_date'] = df['Ride_start_date'].values.astype('datetime64[D]')
df['Boarding_stop_stn'] = df['Boarding_stop_stn'].astype("str")
df['Alighting_stop_stn'] = df['Alighting_stop_stn'].astype('str')
df['Bus_Reg_Num'] = df['Bus_Reg_Num'].astype('str')
df['Bus_Trip_Num'] = df['Bus_Trip_Num'].astype('str')

# Get unique dates
dates = pd.Series.unique(df["Ride_start_date"])
dates = dates.astype('datetime64[D]')
dates = np.sort(dates)

# Get all stops
stops_raw = ["10589", "10017", "05013", "05023", "04222", "04142", "01012", "01113", "01121", "01211", "01311", "80011",
             "80031", "80051", "80071", "80091", "81011", "81031", "81051", "82011", "82032", "82061", "83011", "83031",
             "83062", "83081", "84011", "84021", "84031", "84041", "84051", "84061", "85091", "85051", "85061", "85081",
             "85071", "96071", "96081", "96091", "96041", "96051", "96061", "97011", "97031", "97041", "97051", "97201",
             "97061", "97071", "97081", "97091", "98061", "98071", "99011", "99021", "99031", "99041", "99139", "99009"]
stops = []
for s in stops_raw:
    char_list = list(s)
    while char_list[0] == '0':
        char_list.pop(0)

    code_modified = ''.join(char_list)
    stops.append(code_modified)

df_day = df[df['Ride_start_date'] == dates[0]]
df_day = df[['Bus_Reg_Num', 'Bus_Trip_Num', 'Boarding_stop_stn']]
df_day['uid'] = df[['Bus_Reg_Num', 'Bus_Trip_Num']].apply(lambda x: ' '.join(x), axis=1)
df_day.drop(['Bus_Reg_Num', 'Bus_Trip_Num'], inplace=True, axis=1)

pivot = df_day.pivot_table(index=['uid'], columns='Boarding_stop_stn', aggfunc=len)
count_table = df_day.groupby(['uid', 'Boarding_stop_stn']).size()
time = df_day.drop_duplicates(['uid', 'Boarding_stop_stn'])

new_table = pd.merge(count_table.reset_index(), time, left_on=['uid', 'Boarding_stop_stn'], right_on=['uid', 'Boarding_stop_stn'])

print('Done')
