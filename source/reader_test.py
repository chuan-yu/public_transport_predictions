import reader

def main():
    data_path = "data/count_by_hour_with_header.csv"
    train, val, test = reader.mrt_simple_lstm_data(2, 20, data_path=data_path)
    return

if __name__ == "__main__":
    main()
