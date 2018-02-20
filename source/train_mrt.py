from simple_lstm import Simple_LSTM
from seq2seq import Seq2Seq
import reader
import os
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt


class LSTMConfig():
    def __init__(self, station_code):
        self.train_batch_size = 32
        self.state_size = [200]
        self.feature_len = 3
        self.output_time_steps = 10
        self.input_time_steps = 50
        self.lr = 0.005
        self.num_epochs = 400
        self.keep_prob = 1.0
        self.lr_decay = 1.0
        self.checkpoint = os.path.join("checkpoints/test/mrt_baseline-(200)-lr0.005-no_decay", str(station_code), "checkpoint.ckpt")
        self.write_summary = False
        self.tensorboard_dir = "summaries/lr0.01-decay_0.5_50"


if __name__ == "__main__":

    # HOLIDAYS = ['2016-03-24']
    # HOLIDAYS = [datetime.strptime(h, '%Y-%m-%d') for h in HOLIDAYS]
    # stations = [0, 8, 27, 32, 69, 75, 100, 110, 111]
    stations = [110]
    for s in stations:

        config = LSTMConfig(s)

        # Build model
        lstm_model = Simple_LSTM(config)

        # Load data
        data_path = "data/count_by_hour_with_header.csv"
        data_scaled = reader.get_scaled_mrt_data(data_path, [s], datetime_features=True)
        train, val, test, test_time_features = reader.produce_seq2seq_data(data_scaled,
                                                                           batch_size=config.train_batch_size,
                                                                           input_seq_len=config.input_time_steps,
                                                                           output_seq_len=config.output_time_steps)
        x_train, y_train = train[0], train[1]
        x_val, y_val, = val[0], val[1]
        x_test, y_test = test[0], test[1]

        x_val = np.squeeze(x_val, axis=1)
        x_test = np.squeeze(x_test, axis=1)
        y_train = np.squeeze(y_train, axis=3)
        y_val = np.squeeze(y_val, axis=(1, 3))
        y_test = np.squeeze(y_test, axis=(1, 3))

        # test_time_features = np.squeeze(test_time_features, axis=(1, 2))

        # Run training
        lstm_model.fit(x_train, y_train, x_val, y_val)

        # Make 1-step predictions
        # predictions = lstm_model.predict(x_test[:, 0, :, :])
        # plt.plot(predictions[:, 0], label="predictions")
        # y_true = y_test.reshape((y_test.shape[0], 1))
        # plt.plot(y_true, label="true values")
        # plt.legend(loc='upper right')
        # plt.show()

        # Make multiple-step predictions
        # predictions, rmse = lstm_model.predict(x_test, y_test)
        #
        # print(rmse)
        #
        # plt.plot(data_scaled[:, 0], label="true values")
        # num_test = round(data_scaled.shape[0] * 0.2)
        # plt.plot(range(data_scaled.shape[0]-num_test, data_scaled.shape[0]-num_test+predictions.size), predictions, label="predictions")
        # plt.axvline(x=data_scaled.shape[0]-num_test, color='green', linestyle='--')
        # # plt.title("STN Admiralty")
        # plt.legend(loc='upper right')
        # plt.show()


    ############
    ## Sine data
    ############
    # x = np.linspace(0, 120, 420)
    # data = y = reader.generate_sin_signal(x, noisy=True)
    # train, val, test = reader.mrt_simple_lstm_data(data, config.train_batch_size, config.time_steps)
    # x_train, y_train = train[0], train[1]
    # x_val, y_val, = val[0], val[1]
    # x_test, y_test = test[0], test[1]
    # lstm_model.fit(x_train, y_train, x_val, y_val)
    #
    # # predictions = lstm_model.predict(x_test)
    #
    # x_input = x_test[0, :, :]
    # predictions = lstm_model.predict_multiple_steps(x_input, y_test.shape[0])
    # plt.plot(predictions[:, 0], label="predictions")
    # plt.plot(y_test[:, 0], label="true values")
    # plt.legend(loc='upper right')
    # plt.show()
    #
    # squared_error = np.square(np.subtract(predictions[:, 0], y_test[:, 0]))
    # mean_error = np.mean(squared_error)
    # rmse = np.sqrt(mean_error)
    #
    # print(rmse)