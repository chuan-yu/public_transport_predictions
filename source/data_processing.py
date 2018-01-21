import numpy as np
import pandas as pd
import datetime
from os import listdir, makedirs
from os.path import isfile, join, exists


def load_and_pre_process(file, direction):
    # Load Data
    file_path = file
    df = pd.DataFrame.from_csv(file_path, index_col=None)

    # Select columns
    df = df[['Bus_Trip_Num', 'Bus_Reg_Num', 'Boarding_stop_stn', 'Ride_start_date', 'Ride_start_time', 'Direction']]

    # Convert types
    df['Ride_start_date'] = df['Ride_start_date'].values.astype('datetime64[D]')
    df['Ride_start_time'] = pd.to_datetime(df['Ride_start_time'], format='%H:%M:%S').dt.time
    df['Boarding_stop_stn'] = df['Boarding_stop_stn'].astype("str")
    df['Bus_Reg_Num'] = df['Bus_Reg_Num'].astype('str')
    df['Bus_Trip_Num'] = df['Bus_Trip_Num'].astype('str')

    # Combine date and time
    df['id'] = df[['Bus_Reg_Num', 'Bus_Trip_Num']].apply(lambda x: ' '.join(x), axis=1)
    df.drop(['Bus_Reg_Num', 'Bus_Trip_Num'], inplace=True, axis=1)

    df = df[df["Direction"] == 0]
    df.drop(['Direction'], inplace=True, axis=1)

    return df



def group_and_count(df):
    # Get unique dates
    dates = pd.Series.unique(df["Ride_start_date"])
    dates = dates.astype('datetime64[D]')
    dates = np.sort(dates)

    # Group records by
    result_table = pd.DataFrame()
    for date in dates:
        df_day = df[df['Ride_start_date'] == date]
        count_table = df_day.groupby(['id', 'Boarding_stop_stn']).size()
        time_table = df_day.drop_duplicates(['id', 'Boarding_stop_stn'])
        results = pd.merge(count_table.reset_index(name='count'), time_table, left_on=['id', 'Boarding_stop_stn'], right_on=['id', 'Boarding_stop_stn'])
        results = results.drop('id', axis=1).sort_values(by=['Boarding_stop_stn', 'Ride_start_time'])
        result_table = result_table.append(results)


    result_table['Ride_start_datetime'] = result_table[['Ride_start_date','Ride_start_time']].apply(lambda x: datetime.datetime.combine(*list(x)),axis=1)
    result_table.drop(['Ride_start_date','Ride_start_time'], axis=1, inplace=True)
    result_table['Ride_start_datetime'] = result_table['Ride_start_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    result_table['count'] = result_table['count'].astype('str')

    # index_list = []
    # index_array = result_table[['Boarding_stop_stn', 'Ride_start_datetime']].transpose().as_matrix()
    # index_list.append(index_array[0])
    # index_list.append(index_array[1])
    # count_array = result_table['count'].as_matrix()
    # count_mult = pd.Series(count_array, index=index_list)
    #
    # count_mult.to_csv(RESULT_FILE)

    return result_table


if __name__ == "__main__":

    SERVICE_NO = '2'
    DATA_FOLDER = 'data/' + SERVICE_NO + '/'
    RESULT_FOLDER = 'data/result/'
    DIRECTION = 0

    files = [f for f in listdir(DATA_FOLDER) if isfile(join(DATA_FOLDER, f))]

    if not exists(RESULT_FOLDER):
        makedirs(RESULT_FOLDER)

    for file in files:
        print("processing ", file)
        month_str = file.split('_')[2]
        # print(DATA_FOLDER + file)
        df = load_and_pre_process(DATA_FOLDER + file, DIRECTION)
        result_table = group_and_count(df)
        result_file = RESULT_FOLDER +  SERVICE_NO + '_' + month_str + '_' + str(DIRECTION) + '.csv'
        result_table.to_csv(result_file, index=False)







