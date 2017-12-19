import reader

def main():
    data_path = "data/count_by_hour_with_header.csv"
    x, y = reader.get_raw_data(data_path)
    return

if __name__ == "__main__":
    main()
