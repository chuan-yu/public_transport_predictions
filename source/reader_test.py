import reader
import numpy as np
import matplotlib.pyplot as plt

def main():
    file = 'data/result/2APR_0.csv'
    df = reader.get_scaled_bus_data(['84031'], file, True)
    # plt.plot(df)
    # plt.show()

    train, val, test, time_features = reader.produce_seq2seq_data(df, 32, 50, 10)
    x_train, y_train = train[0], train[1]
    x_val, y_val = val[0], val[1]
    x_test, y_test = test[0], test[1]


    return

if __name__ == "__main__":
    main()
