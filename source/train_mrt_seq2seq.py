from seq2seq import Seq2Seq
import reader
from configs.configs import Seq2SeqConfig
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # stations = [0, 8, 27, 32, 69, 75, 100, 110, 111]
    stations = [32]

    HOLIDAYS = None #['2016-03-25']

    for s in stations:
        print("station %i..." % s)
        config = Seq2SeqConfig(s)
        model = Seq2Seq(config)

        # Load data
        data_path = "data/count_by_hour_with_header.csv"
        data_scaled = reader.get_scaled_mrt_data(data_path, [s], datetime_features=True, holidays=HOLIDAYS)

        n = data_scaled.shape[0]
        x_pretrain, y_pretrain = reader.get_pretrain_data(data_scaled[0:round(0.6 * n)],
                                                          3, config.input_time_steps)

        # model.pretrain_encoder(x_pretrain, y_pretrain, 0.001, 2000)
        # model = Seq2Seq(config, False)
        train, val, test = reader.produce_seq2seq_data(data_scaled, config.train_batch_size, config.input_time_steps,
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


        model.fit(x_train, targets_train, features_train, x_val, targets_val, features_val)

        predictions, rmse = model.predict(x_test, targets_test, features_test)
        print(rmse)
        #
        # plt.plot(data_scaled[:, 0], label="true values")
        # num_test = round(data_scaled.shape[0] * 0.2)
        # plt.plot(range(data_scaled.shape[0]-num_test, data_scaled.shape[0]-num_test+predictions.size), predictions, label="predictions")
        # plt.axvline(x=data_scaled.shape[0]-num_test, color='green', linestyle='--')
        # # plt.title("STN Admiralty")
        # plt.legend(loc='upper right')
        # plt.show()