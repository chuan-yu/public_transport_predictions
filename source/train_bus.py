from simple_lstm import Simple_LSTM
import reader
import os
import numpy as np
from matplotlib import pyplot as plt

class LSTMConfig():
    def __init__(self, stop_code):
        self.train_batch_size = 30
        self.state_size = 30
        self.num_layers = 1
        self.input_size = 4
        self.output_size = 1
        self.time_steps = 50
        self.lr = 0.001
        self.num_epochs = 400
        self.checkpoint = os.path.join("checkpoints/bus", str(stop_code), "checkpoint.ckpt")
        self.tensorboard_dir = "summaries/"

if __name__ == "__main__":

    stops = ['84031']

    for s in stops:
        config = LSTMConfig(s)

        # Build model
        lstm_model = Simple_LSTM(config)

        data_file = "data/result/2APR_0.csv"
        df = reader.get_scaled_bus_data(['84031'], data_file, True)

        train, val, test, test_time_features = reader.produce_seq2seq_data(df, config.train_batch_size,
                                                                           config.time_steps,
                                                                           output_seq_len=1)
        x_train, y_train = train[0], train[1]
        x_val, y_val, = val[0], val[1]
        x_test, y_test = test[0], test[1]

        x_val = np.squeeze(x_val, axis=1)
        x_test = np.squeeze(x_test, axis=1)
        y_train = np.squeeze(y_train, axis=2)
        y_val = np.squeeze(y_val, axis=(1, 2))
        y_test = np.squeeze(y_test, axis=(1, 2))

        test_time_features = np.squeeze(test_time_features, axis=(1, 2))

        # Run training
        lstm_model.fit(x_train, y_train, x_val, y_val)