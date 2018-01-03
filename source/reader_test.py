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
    res = reader.produce_seq2seq_data(y, 2, 5, 3)
    return

if __name__ == "__main__":
    main()
