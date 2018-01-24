from simple_lstm import Simple_LSTM
import reader
import os
import numpy as np
from matplotlib import pyplot as plt


class LSTMConfig():
    def __init__(self, station_code):
        self.train_batch_size = 30
        self.state_size = 30
        self.num_layers = 2
        self.input_size = 3
        self.output_size = 1
        self.time_steps = 50
        self.lr = 0.0005
        self.num_epochs = 400
        self.checkpoint = os.path.join("checkpoints/lstm_without_time_features_lr0.001", str(station_code), "checkpoint.ckpt")
        self.tensorboard_dir = "summaries/"


if __name__ == "__main__":


    #########
    ## MRT
    #########

    # stations = [0, 8, 69, 75, 100, 110]
    stations = [0]
    # stations =[100]

    for s in stations:

        config = LSTMConfig(s)

        # Build model
        lstm_model = Simple_LSTM(config)

        # Load data
        data_path = "data/count_by_hour_with_header.csv"
        data_scaled = reader.get_scaled_mrt_data(data_path, [s], datetime_features=True)
        train, val, test, test_time_features = reader.produce_seq2seq_data(data_scaled, config.train_batch_size, config.time_steps,
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

        # Make 1-step predictions
        # predictions = lstm_model.predict(x_test[:, 0, :, :])
        # plt.plot(predictions[:, 0], label="predictions")
        # y_true = y_test.reshape((y_test.shape[0], 1))
        # plt.plot(y_true, label="true values")
        # plt.legend(loc='upper right')
        # plt.show()

        # Make multiple-step predictions
        x_input = x_test[0]
        predictions, rmse = lstm_model.predict_multiple_steps(x_input, test_time_features, y_test)

        print(rmse)

        plt.plot(data_scaled[:, 0], label="true values")
        plt.plot(range(data_scaled.shape[0]-y_test.shape[0], data_scaled.shape[0]), predictions, label="predictions")
        plt.axvline(x=594, color='green', linestyle='--')
        # plt.title("STN Admiralty")
        plt.legend(loc='upper right')
        plt.show()


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