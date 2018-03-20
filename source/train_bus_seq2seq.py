import numpy as np
from seq2seq import Seq2Seq
from configs.configs import Seq2SeqConfigBus
import reader
from matplotlib import pyplot as plt

if __name__ == "__main__":

    DATA_LIST_FILE = "../data/stop_boarding_counts/bus_no_2/data_file_list.txt"

    stops = ['1113', '5013', '83062', '84031', '85091']
    # stops = ['84031']

    for s in stops:
        print("Stop", s)
        config = Seq2SeqConfigBus(s)

        # Build model
        lstm_model = Seq2Seq(config)

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

        train, val, test = reader.produce_seq2seq_data(df, config.train_batch_size, config.input_time_steps,
                                                       config.output_time_steps, time_major=True, y_has_features=True)

        x_train, y_train = train[0], train[1]
        x_val, y_val, = val[0], val[1]
        x_test, y_test = test[0], test[1]

        targets_train = y_train[:, :, :, [0]]
        features_train = y_train[:, :, :, 1:]

        targets_val = y_val[:, :, :, [0]]
        features_val = y_val[:, :, :, 1:]

        targets_test = y_test[:, :, :, [0]]
        features_test = y_test[:, :, :, 1:]


        # Run training
        # lstm_model.fit(x_train, targets_train, features_train, x_val, targets_val, features_val)

        # x_input = x_test[0]
        # predictions, rmse = lstm_model.predict_multiple_steps(x_input, test_time_features, y_test)

        predictions, rmse = lstm_model.predict(x_test, targets_test, features_test)
        print(rmse)

        # plt.plot(y_test.flatten(), label="true values")
        # plt.plot(predictions, label="predictions")
        # plt.legend(loc='upper right')
        # plt.show()