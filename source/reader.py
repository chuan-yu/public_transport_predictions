import numpy as np
import pandas as pd

def _read_raw_data(data_path=None):
    '''Read data from a csv file.

    :param data_path: the path of data file.
    :return: pandas data frame of size [time_steps, num_stations].

    '''

    data = pd.read_csv(data_path, index_col=0)
    data.reset_index(drop=True, inplace=True)
    data = pd.DataFrame.transpose(data)
    return data


def _scale(data):
    '''Rescale the data by column to the range between 0 and 1.

    For each column, X = (X - X.min()) / (X.max() - X.min())

    :param data: numpy 2-d darray
    :return: numpy 2-d array
    '''

    data_mins = data.min(axis=0)
    data_mins = data_mins.reshape((-1, data_mins.shape[0]))
    data = np.subtract(data, data_mins)
    minmax_range = data.max(axis=0) - data.min(axis=0)
    minmax_range = minmax_range.reshape((-1, minmax_range.shape[0]))
    data = np.divide(data, minmax_range)
    return data

def get_scaled_mrt_data(data_path=None):
    ''' Get scaled mrt data

    :param data_path: file path of the data
    :return: numpy array of size [time_steps, num_stations]
    '''
    # read raw data as pandas Dataframe
    raw_data = _read_raw_data(data_path)

    # # Convert the column name to datetime and offset the hour by 1
    # # Offset by 1 because the daytime is for the next time step
    # date_time = pd.to_datetime(raw_data.columns.values) + pd.Timedelta(hours=1)
    #
    # # Get hour from column name
    # hour = np.array(date_time.hour.tolist())
    # hour = hour.reshape((1, hour.shape[0]))
    # hour = _scale(hour)
    #
    # # Get day of week from column name
    # day_week = np.array(date_time.dayofweek.tolist())
    # day_week = day_week.reshape((1, day_week.shape[0]))
    # day_week = _scale(day_week)

    data = raw_data.as_matrix(columns=None)
    data = _scale(data)

    return data


def mrt_simple_lstm_data(batch_size, truncated_backpro_len,
                         train_ratio=0.6, val_ratio=0.2, data_path=None):
    '''Produce the training, validation and test set

    :param batch_size: training data batch size
    :param truncated_backpro_len: the number of time steps with which backpropogation through time is done. Also equal to
    the time window size.
    :param num_prediction_steps: the number of time steps in the future to be predicted.
    :param train_ratio: the ratio of training samples to the total data samples. The value is a float between 0 and 1.
    Default is 0.6.
    :param val_ratio: the ration of training samples to the total data samples. The value is a float between 0 and 1.
    Default is 0.2.
    :param data_path: the file path to the data file
    :return:
            (x_train, y_train), (x_val, y_val), (x_test, y_test)
            x_train: [num_train_batches, batch_size, truncated_backpro_len, feature_len],
            y_train: [num_train_batches, batch_size, feature_len],
            x_val: [num_val_samples, truncated_backpro_len, feature_len]
            y_val: [num_val_samples, feature_len]
            x_test: [num_test_samples, truncated_backpro_len, feature_len]
            y_test: [num_test_samples, feature_dim]
    '''

    data = get_scaled_mrt_data(data_path)
    total_data_len = data.shape[0]
    num_features = data.shape[1]

    seq_len = truncated_backpro_len + 1
    num_time_windows = total_data_len - seq_len
    time_windows = []

    for i in range(num_time_windows):
        time_windows.append(data[i:i + seq_len, :])

    time_windows = np.array(time_windows)

    num_train = round(num_time_windows * train_ratio)
    train = time_windows[0:num_train, :, :]

    num_val = round(num_time_windows * val_ratio)
    val = time_windows[num_train:num_train + num_val, :, :]

    test = time_windows[num_train + num_val:, :, :]

    num_train_batches = num_train // batch_size
    train = train[0:num_train_batches *  batch_size, :, :]

    train = np.reshape(train, (-1, batch_size, train.shape[1], train.shape[2]))
    x_train = train[:, :, :-1, :]
    y_train = train[:, :, -1, :]

    x_val = val[:, :-1, :]
    y_val = val[:, -1, :]

    x_test = test[:, :-1, :]
    y_test = test[:, -1, :]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# def lstm_data_producer(batch_size, data_path=None):
#     train_portions = 0.6
#     val_portions = 0.2
#
#     # Get raw data
#     x, y = get_raw_data(data_path)
#
#     # split train, val, test
#     total_time_steps = x.shape[0]
#     train_time_steps = int(total_time_steps * train_portions)
#     val_time_steps = int(total_time_steps * val_portions)
#     x_train = x[0:train_time_steps, :]
#     y_train = y[0:train_time_steps, :]
#     x_val = x[train_time_steps:train_time_steps+val_time_steps, :]
#     y_val = y[train_time_steps:train_time_steps+val_time_steps, :]
#     x_test = x[train_time_steps+val_time_steps:, :]
#     y_test = y[train_time_steps + val_time_steps:, :]
#
#     # reshape to [batch_size, input_size]
#     train_batch_len = x_train.shape[0] // batch_size
#     x_train = x_train[0:batch_size * train_batch_len]
#     y_train = y_train[0:batch_size * train_batch_len]
#
#     x_train = np.reshape(x_train, (batch_size, -1, x_train.shape[1]))
#     y_train = np.reshape(y_train, (batch_size, -1, y_train.shape[1]))
#
#     x_val = np.reshape(x_val, (1, -1, x_val.shape[1]))  # batch_size=1 for val set
#     y_val = np.reshape(y_val, (1, -1, y_val.shape[1]))
#
#     x_test = np.reshape(x_test, (1, -1, x_test.shape[1]))  # batch_size=1 for test set
#     y_test = np.reshape(y_test, (1, -1, y_test.shape[1]))
#
#     return (x_train, y_train), (x_val, y_val), (x_test, y_test)






