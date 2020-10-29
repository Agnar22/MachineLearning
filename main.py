import pandas as pd
import config
import matplotlib.pyplot as plt


def load_and_clean_data():
    data = pd.read_csv(config.DATA_PATH)
    data['date'] = pd.DatetimeIndex(data['date'])
    data = data[data['location'].isin(config.COUNTRIES)]
    data = data.fillna(0)
    return data


def draw_graph(*args, x: str = 'x', y: str = 'y'):
    """
    :param y: label for y axis.
    :param x: label for x axis.
    :param args: dict('x' : list, 'y' : list, 'name' : str)
    :return:
    """

    plt.close('all')
    for func in args:
        plt.plot(func['x'], func['y'], label=func['name'])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(fontsize=8)
    plt.legend()
    plt.show()


def visualize_spread_for_countries(data: pd.DataFrame):
    """
    :param data: a pandas dataframe of the data to visualize.
    :return:
    """
    countries_to_visualize = []
    for country in config.COUNTRIES:
        countries_to_visualize.append(
            {
                'x': data[data['location'] == country]['date'],
                'y': data[data['location'] == country]['total_cases_per_million'],
                'name': country
            }
        )
    draw_graph(*countries_to_visualize, x='date', y='total cases per million')


if __name__ == '__main__':
    data = load_and_clean_data()
    print(data.columns)
    visualize_spread_for_countries(data)
