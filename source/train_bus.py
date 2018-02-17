from simple_lstm import Simple_LSTM
import reader
import os
import numpy as np
from matplotlib import pyplot as plt

DATA_LIST_FILE = "../data/stop_boarding_counts/bus_no_2/data_file_list.txt"

class LSTMConfig():
    def __init__(self, stop_code):
        self.train_batch_size = 128
        self.state_size = [200, 200]
        self.feature_len = 4
        self.output_time_steps = 10
        self.input_time_steps = 50
        self.lr = 0.005
        self.lr_decay = 1.0
        self.num_epochs = 200
        self.keep_prob = 0.5
        self.write_summary = False
        self.checkpoint = os.path.join("../checkpoints/bus", stop_code, "checkpoint.ckpt")
        self.tensorboard_dir = "summaries/bus"

if __name__ == "__main__":

    stops = ['1113', '5013', '83062', '84031', '85091']
    # stops = ['83062', '84031', '85091']

    for s in stops:
        print("Stop", s)
        config = LSTMConfig(s)

        # Build model
        lstm_model = Simple_LSTM(config)

        with open(DATA_LIST_FILE) as f:
            data_file_list = f.readlines()
        data_file_list = [fname.rstrip('\n') for fname in data_file_list]

        df = []

        for file in data_file_list:
            print("Loading", file)
            data = reader.get_scaled_bus_data([s], "../" + file, '30T', True)
            df.append(data)

        df = np.array(df)
        df = np.vstack(df)

        train, val, test, test_time_features = reader.produce_seq2seq_data(df,
                                                                           batch_size=config.train_batch_size,
                                                                           input_seq_len=config.input_time_steps,
                                                                           output_seq_len=config.output_time_steps)

        x_train, y_train = train[0], train[1]
        x_val, y_val, = val[0], val[1]
        x_test, y_test = test[0], test[1]

        # Shuffle training data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(x_train.shape[0]))
        x_train = x_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        x_val = np.squeeze(x_val, axis=1)
        x_test = np.squeeze(x_test, axis=1)
        y_train = np.squeeze(y_train, axis=3)
        y_val = np.squeeze(y_val, axis=(1, 3))
        y_test = np.squeeze(y_test, axis=(1, 3))

        # test_time_features = np.squeeze(test_time_features, axis=(1, 2))

        # Run training
        # lstm_model.fit(x_train, y_train, x_val, y_val)

        # x_input = x_test[0]
        # predictions, rmse = lstm_model.predict_multiple_steps(x_input, test_time_features, y_test)

        predictions, rmse = lstm_model.predict(x_test, y_test)
        print(rmse)

        plt.plot(y_test.flatten(), label="true values")
        plt.plot(predictions, label="predictions")
        plt.legend(loc='upper right')
        plt.show()