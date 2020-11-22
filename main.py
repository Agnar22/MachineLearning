import pandas as pd
import config
import matplotlib.pyplot as plt
import numpy as np
import lstm
from sklearn.model_selection import train_test_split


def groups_to_cases(groups, overlapping: bool = False):
  """
  :param groups:
  :param overlapping:
  :return:
  """
  y = np.array([])
  x = np.array([]).reshape(-1, config.INPUTDAYS)
  for _, group in groups:
    x_group, y_group = group_to_cases(group, overlapping=overlapping)
    y = np.concatenate((y, y_group))
    x = np.concatenate((x, x_group))
  return x, y


def group_to_cases(group, overlapping: bool = False):
  """
  :param group:
  :param overlapping:
  :return:
  """

  if overlapping:
    y = group['ConfirmedCases'].iloc[config.INPUTDAYS:].to_numpy()
    x = np.array([]).reshape(-1, config.INPUTDAYS)
    for row in range(group.shape[0] - config.INPUTDAYS):
      curr_x = group['ConfirmedCases'].iloc[row:row + config.INPUTDAYS].to_numpy()
      x = np.concatenate((x, curr_x.reshape(1, config.INPUTDAYS)), axis=0)
  else:
    y = group['ConfirmedCases'].iloc[config.INPUTDAYS::config.INPUTDAYS + 1].to_numpy()
    x = group['ConfirmedCases'].to_numpy().reshape(-1, config.INPUTDAYS + 1)
    x = np.delete(x, config.INPUTDAYS, 1)
  return x, y


def create_supervised_data_set(data: pd.DataFrame, overlapping: bool = False):
  """
  :param data:
  :return: supervised data set (input and target)
  """
  data = data[data['ConfirmedCases'] > config.INFECTED_LOWER].groupby('CountryName')
  x, y = groups_to_cases(data, overlapping=overlapping)
  return x, y


def normalize_data(x: np.ndarray, y: np.ndarray = None):
  """
  :param x:
  :param y:
  :return: normalized values for x
  """

  dim = len(x.shape) - 1

  x_max, x_min = (x.max(axis=dim, keepdims=True), x.min(axis=dim, keepdims=True))
  difference = np.clip(x_max - x_min, 0.000001, None)

  # Normalize y.
  if (y is not None):
    y = y.reshape((-1, 1))
    y = (2 * (y - x.min(axis=dim, keepdims=True))) / difference - 1

  # Normalize the values from -1 to 1.
  x = (2 * (x - x.min(axis=dim, keepdims=True))) / difference - 1

  # Drop values that are normalized wrong.
  # for row, val in enumerate(np.where(y > 10)):
  #  print("over 10", row, val, x[val], y[val])

  if (y is not None):
    invalid_rows = np.where((y > 2) | (y < 1))
    x_max = np.delete(x_max, invalid_rows, axis=0)
    x_min = np.delete(x_min, invalid_rows, axis=0)
    x = np.delete(x, invalid_rows, axis=0)
    y = np.delete(y, invalid_rows, axis=0)

  return x, y, x_max, x_min


# Split dataset into testing & training
def split_data(x: np.ndarray, y: np.ndarray):
  """
  :param x:
  :param y:
  :return:
  """
  X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=config.VALIDATION_SIZE, random_state=2)
  X_train = X_train.reshape(*X_train.shape, 1)
  X_test = X_test.reshape(*X_test.shape, 1)
  return X_train, X_test, Y_train, Y_test


def load_and_clean_data():
  """
  :return:
  """
  data = pd.read_csv(config.DATA_PATH)
  data['date'] = pd.DatetimeIndex(data['Date'])
  data = data.fillna(0)
  return data


def draw_graph(*args, x: str = 'x', y: str = 'y'):
  """
  :param args: dict('x' : list, 'y' : list, 'name' : str)
  :param y: label for y axis.
  :param x: label for x axis.
  :return:
  """

  plt.close('all')
  for func in args:
    plt.plot(func['x'], func['y'], label=func['name'])
  plt.xlabel(x)
  plt.ylabel(y)
  plt.xticks(fontsize=8)
  plt.legend()
  plt.show(block=False)


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


def visualize_predictions(cases: pd.DataFrame):
  loop = True
  while loop:
    try:
      start_day = int(input("Start day:"))
      prediction_length = int(input("prediction length:"))
      output_start = start_day + config.INPUTDAYS
      output_end = output_start + prediction_length

      predictions = lstm.predict(model, cases.iloc[start_day:output_start]['ConfirmedCases'].to_numpy(),
                                 prediction_length)
      for day in range(predictions.shape[0]):
        print(data['date'].iloc[output_start + day], predictions[day], cases.iloc[output_start + day]['ConfirmedCases'])
      draw_graph(
        {'x': cases['date'].iloc[output_start:output_end], 'y': predictions.tolist(), 'name': 'prediction'},
        {'x': cases['date'].iloc[:start_day], 'y': cases_norway['ConfirmedCases'].iloc[:start_day], 'name': 'start'},
        {'x': cases['date'].iloc[start_day:output_start], 'y': cases['ConfirmedCases'].iloc[start_day:output_start],
         'name': 'input'},
        {'x': cases['date'].iloc[output_start:output_end], 'y': cases['ConfirmedCases'].iloc[output_start:output_end],
         'name': 'target'},
      )
    except:
      ans = input("quit?")
      if ans == 'y':
        loop = False


def de_normalize(x: np.ndarray, x_max: np.ndarray, x_min: np.ndarray):
  return (x + 1) * (x_max - x_min) / 2 + x_min


if __name__ == '__main__':
  data = load_and_clean_data()
  # visualize_spread_for_countries(data)

  x, y = create_supervised_data_set(data[data['CountryName'] != 'Norway'], overlapping=True)
  x_norm, y_norm, x_max, x_min = normalize_data(x, y)
  X_train, X_test, Y_train, Y_test = split_data(x_norm, y_norm)

  model = lstm.create_model()
  train_hist = lstm.train_model(model, X_train, Y_train, validation=(X_test, Y_test))
  draw_graph(
    {'x': train_hist.index, 'y': train_hist['loss'], 'name': 'training'},
    {'x': train_hist.index, 'y': train_hist['val_loss'], 'name': 'validation'}
  )
  predictions = model.predict(X_train)
  loss = (predictions - Y_train) ** 2
  # plt.boxplot(Y_test)
  # plt.show()

  # for pos in np.flip(np.argsort(loss, axis=0))[0:10]:
  #  print(pos, X_train[pos], Y_train[pos], predictions[pos], loss[pos])

  cases_norway = data[data['location'] == 'Norway']
  visualize_predictions(cases_norway)
