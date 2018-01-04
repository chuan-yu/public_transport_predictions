import reader
import numpy as np
import matplotlib.pyplot as plt

def main():
    # # data_path = "data/count_by_hour_with_header.csv"
    # # data_scaled = reader.get_scaled_mrt_data(data_path=data_path)
    # # train, val, test = reader.mrt_simple_lstm_data(data_scaled, 2, 20)
    x = np.linspace(0, 30, 105)
    y = reader.generate_sin_signal(x, noisy=False)
    # plt.plot(x, y, 'ro')
    # plt.show()
    train, val, test = reader.produce_seq2seq_data(y, 2, 5, 3, time_major=False)
    x_train, y_train = train[0], train[1]
    x_val, y_val = val[0], val[1]
    x_test, y_test = test[0], test[1]

    input = y_test[2, 0, :, :]
    output = y_test[3, 0, :, :]

    plt.plot(range(3), input, 'ro')
    plt.plot(range(3, 6), output, 'bo')
    plt.show()

    return

if __name__ == "__main__":
    main()
