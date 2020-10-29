import pandas as pd
import config


def load_and_clean_data():
    data = pd.read_csv(config.DATA_PATH)
    data = data[data['location'].isin(config.COUNTRIES)]
    data = data.fillna(0)
    return data


if __name__ == '__main__':
    print(load_and_clean_data()['total_cases'])
