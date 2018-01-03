import numpy as np
import pandas as pd


def generate_sin_signal(x, noisy=False):
    y = np.sin(x)
    if noisy:
        noise = np.random.rand(len(x)) * 0.2
        y = y + noise

    y = np.reshape(y, (y.shape[0], 1))
    return y


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

    :param data: numpy 2-d array
    :return: numpy 2-d array
    '''

    data_mins = data.min(axis=0)
    data_mins = data_mins.reshape((-1, data_mins.shape[0]))
    data = np.subtract(data, data_mins)
    minmax_range = data.max(axis=0) - data.min(axis=0)
    minmax_range = minmax_range.reshape((-1, minmax_range.shape[0]))
    data = np.divide(data, minmax_range)
    return data

def get_scaled_mrt_data(data_path=None, stations_codes=None):
    ''' Get scaled mrt data

    :param stations_codes: a list of selected station codes. If None, return data for all stations
    :param data_path: file path of the data
    :return: numpy array of size [time_steps, num_stations]
    '''
    # read raw data as pandas Dataframe
    raw_data = _read_raw_data(data_path)
    if stations_codes is not None:
        raw_data = raw_data[stations_codes]

    data = raw_data.as_matrix(columns=None)
    data = _scale(data)
    return data


def mrt_simple_lstm_data(data, batch_size, truncated_backpro_len,
                         train_ratio=0.6, val_ratio=0.2):
    '''Produce the training, validation and test set

    :param data: time series data of shape [time_steps, feature_len]
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

def produce_seq2seq_data(data, batch_size, input_seq_len, output_seq_len, train_ratio=0.6, val_ratio=0.2):

    total_time_steps = data.shape[0]

    num_train = round(total_time_steps * train_ratio)
    num_val = round(total_time_steps * val_ratio)
    num_test = total_time_steps - num_train - num_val

    train_raw = data[0:num_train]
    val_raw = data[num_train-input_seq_len:num_train + num_val]
    test_raw = data[num_train + num_val-input_seq_len:]

    total_seq_len = input_seq_len + output_seq_len

    num_train_batches = num_train // batch_size
    train_raw = train_raw[0:num_train_batches * batch_size]
    train_raw = np.reshape(train_raw, (batch_size, -1, train_raw.shape[1]))
    val_raw = np.reshape(val_raw, (1, val_raw.shape[0], val_raw.shape[1]))
    test_raw = np.reshape(test_raw, (1, test_raw.shape[0], test_raw.shape[1]))

    train = _convert_to_windows(train_raw, total_seq_len)
    val = _convert_to_windows(val_raw, total_seq_len, False, output_seq_len)
    test = _convert_to_windows(test_raw, total_seq_len, False, output_seq_len)

    train = np.swapaxes(train, 1, 2)
    val = np.swapaxes(val, 1, 2)
    test = np.swapaxes(test, 1, 2)

    x_train = train[:, :-output_seq_len, :, :]
    y_train = train[:, -output_seq_len:, :, :]

    x_val = val[:, :-output_seq_len, :, :]
    y_val = val[:, -output_seq_len:, :, :]

    x_test = test[:, :-output_seq_len, :, :]
    y_test = test[:, -output_seq_len:, :, :]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def _convert_to_windows(data, total_seq_len, train=True, output_seq_len=1):
    '''Convert time series data to time windows of seq_len
    Example:
        data = [0, 1, 2, 3, 4, 5, 6]
        total_seq_len = 5
        output_seq_len = 2
        if train:
            return [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
        if not train:
            return [[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]]
            when train is False, time windows are taken by shifting output_seq_len positions to the right

    :param data: input data of shape [batch_size, time_steps, feature_len]
    :param total_seq_len: sequence length
    :param output_seq_len: output sequence length
    :param train: whether the output is for training
    :return: numpy array of shape [num_time_windows, batch_size, total_seq_len, feature_len]
    '''
    if train:
        forward_steps = 1
    else:
        forward_steps = output_seq_len

    num_time_windows = (data.shape[1] - total_seq_len) // forward_steps + 1
    time_windows = np.ndarray((num_time_windows, data.shape[0], total_seq_len, data.shape[2]))

    for i in range(num_time_windows):
        time_windows[i] = data[:, i * forward_steps:i * forward_steps + total_seq_len]

    return time_windows






