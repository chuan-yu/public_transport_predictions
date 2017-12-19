import numpy as np
import pandas as pd

def _read_raw_data(data_path=None):
    data = pd.read_csv(data_path, index_col=0)
    data.reset_index(drop=True, inplace=True)
    return data

# Rescale the data to the range between 0 and 1
# Scaling is done by row
# For each row, X = (X - X.min) / (X.max() - X.min())
def _scale(data):
    data_mins = data.min(axis=1)
    data_mins = data_mins.reshape((data_mins.shape[0], -1))
    data = np.subtract(data, data_mins)
    minmax_range = data.max(axis=1) - data.min(axis=1)
    minmax_range = minmax_range.reshape((minmax_range.shape[0], -1))
    data = np.divide(data, minmax_range)
    return data

def get_raw_data(data_path=None):
    # read raw data as pandas Dataframe
    raw_data = _read_raw_data(data_path)

    # Convert the column name to datetime and offset the hour by 1
    # Offset by 1 because the daytime is for the next time step
    date_time = pd.to_datetime(raw_data.columns.values) + pd.Timedelta(hours=1)

    # Get hour from column name
    hour = np.array(date_time.hour.tolist())
    hour = hour.reshape((1, hour.shape[0]))
    hour = _scale(hour)

    # Get day of week from column name
    day_week = np.array(date_time.dayofweek.tolist())
    day_week = day_week.reshape((1, day_week.shape[0]))
    day_week = _scale(day_week)

    data = raw_data.as_matrix(columns=None)
    data = _scale(data)

    # Get label y
    y = np.roll(data, -1, axis=1)
    y = np.delete(y, -1, 1) # drop the last column

    # Get x
    data = np.append(data, hour, axis=0)
    data = np.append(data, day_week, axis=0)
    x = np.delete(data, -1, 1)  # drop last column

    x = np.transpose(x)  # [num_time_steps, num_features]
    y = np.transpose(y)  # [num_time_steps, num_targets]

    return x, y

def lstm_data_producer(batch_size, data_path=None):
    train_portions = 0.6
    val_portions = 0.2

    # Get raw data
    x, y = get_raw_data(data_path)

    # split train, val, test
    total_time_steps = x.shape[0]
    train_time_steps = int(total_time_steps * train_portions)
    val_time_steps = int(total_time_steps * val_portions)
    x_train = x[0:train_time_steps, :]
    y_train = y[0:train_time_steps, :]
    x_val = x[train_time_steps:train_time_steps+val_time_steps, :]
    y_val = y[train_time_steps:train_time_steps+val_time_steps, :]
    x_test = x[train_time_steps+val_time_steps:, :]
    y_test = y[train_time_steps + val_time_steps:, :]

    # reshape to [batch_size, input_size]
    train_batch_len = x_train.shape[0] // batch_size
    x_train = x_train[0:batch_size * train_batch_len]
    y_train = y_train[0:batch_size * train_batch_len]

    x_train = np.reshape(x_train, (batch_size, -1, x_train.shape[1]))
    y_train = np.reshape(y_train, (batch_size, -1, y_train.shape[1]))

    x_val = np.reshape(x_val, (1, -1, x_val.shape[1]))  # batch_size=1 for val set
    y_val = np.reshape(y_val, (1, -1, y_val.shape[1]))

    x_test = np.reshape(x_test, (1, -1, x_test.shape[1]))  # batch_size=1 for test set
    y_test = np.reshape(y_test, (1, -1, y_test.shape[1]))

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)






