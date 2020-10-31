import pandas as pd
import config
import matplotlib.pyplot as plt
import numpy as np
import lstm
from sklearn.model_selection import train_test_split


def create_supervised_data_set(data: pd.DataFrame):
  """
  :param data:
  :return: supervised data set (input and target)
  """
  data = data.groupby('location').apply(
    lambda x: x[x['total_cases'] >= config.INFECTED_LOWER]
  ).apply(
    lambda x: x.take([x for x in range(x.shape[0] - (x.shape[0] % (config.INPUTDAYS + 1)))])
  ).reset_index(drop=True)['total_cases'].to_numpy()

  y = data[config.INPUTDAYS::(config.INPUTDAYS + 1)]
  x = data.reshape((-1, config.INPUTDAYS + 1))
  x = np.delete(x, config.INPUTDAYS, 1)

  return x, y


def normalize_data(x: np.ndarray, y: np.ndarray):
  """
  :param x:
  :param y:
  :return: normalized values for x
  """
  difference = np.clip(x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True), 0.000001, None)

  # Normalize y.
  y = y.reshape((-1, 1))
  y = (2 * (y - x.min(axis=1, keepdims=True))) / difference - 1

  # Normalize the values from -1 to 1.
  x = (2 * (x - x.min(axis=1, keepdims=True))) / difference - 1
  return x, y


# Split dataset into testing & training
def split_data(x: np.ndarray, y: np.ndarray):
  """
  :param x:
  :param y:
  :return:
  """
  X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=config.VALIDATION_SIZE, random_state=42)
  X_train = X_train.reshape(*X_train.shape, 1)
  X_test = X_test.reshape(*X_test.shape, 1)
  return X_train, X_test, Y_train, Y_test


def load_and_clean_data():
  """
  :return:
  """
  data = pd.read_csv(config.DATA_PATH)
  data['date'] = pd.DatetimeIndex(data['date'])
  # data = data[data['location'].isin(config.COUNTRIES)]
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
  visualize_spread_for_countries(data)
  x, y = create_supervised_data_set(data)
  x, y = normalize_data(x, y)
  X_train, X_test, Y_train, Y_test = split_data(x, y)
  print(X_test)
  model = lstm.create_model()
  lstm.train_model(model, X_train, Y_train, validation=(Y_train, Y_test))
