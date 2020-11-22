import pandas as pd
import config
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
  x = np.array([]).reshape(-1, config.INPUTDAYS, 18)
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

  features = ['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events', 'C4_Restrictions on gatherings',
              'C5_Close public transport', 'C6_Stay at home requirements', 'C7_Restrictions on internal movement',
              'C8_International travel controls', 'E1_Income support', 'E2_Debt/contract relief', 'E3_Fiscal measures',
              'E4_International support', 'H1_Public information campaigns', 'H2_Testing policy', 'H3_Contact tracing',
              'H4_Emergency investment in healthcare', 'H6_Facial Coverings', 'ConfirmedCases']

  y = group['ConfirmedCases'].iloc[config.INPUTDAYS:].to_numpy()
  x = np.array([]).reshape(-1, config.INPUTDAYS, len(features))
  for row in range(group.shape[0] - config.INPUTDAYS):
    curr_x = group[features].iloc[row:row + config.INPUTDAYS].to_numpy()
    x = np.concatenate((x, curr_x.reshape(1, config.INPUTDAYS, len(features))), axis=0)
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
  return X_train, X_test, Y_train, Y_test


def load_and_clean_data():
  """
  :return:
  """
  data = pd.read_csv(config.DATA_PATH)
  data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')

  # Remove flags.
  data = data.drop(list(filter(lambda x: 'flag' in x.lower(), data.columns)), axis=1)

  data = data.drop(['M1_Wildcard'], axis=1)

  # Keep relevant columns.
  data = data[list(filter(lambda x: '_' in x or x in ['CountryName', 'Date', 'ConfirmedCases'], data.columns))]

  # TODO: check if this is correct.
  data = data.fillna(0)
  return data



def de_normalize(x: np.ndarray, x_max: np.ndarray, x_min: np.ndarray):
  return (x + 1) * (x_max - x_min) / 2 + x_min


if __name__ == '__main__':
  data = load_and_clean_data()
  data = data.iloc[0:200]
  # visualize_spread_for_countries(data)

  x, y = create_supervised_data_set(data[data['CountryName'] != 'Norway'], overlapping=True)
  x_norm, y_norm = x, y
  X_train, X_test, Y_train, Y_test = split_data(x_norm, y_norm)

  model = lstm.create_model()
  train_hist = lstm.train_model(model, X_train, Y_train, validation=(X_test, Y_test))
  predictions = model.predict(X_train)
  loss = (predictions - Y_train) ** 2
  print("predictions", predictions)
  features = ['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events', 'C4_Restrictions on gatherings',
              'C5_Close public transport', 'C6_Stay at home requirements', 'C7_Restrictions on internal movement',
              'C8_International travel controls', 'E1_Income support', 'E2_Debt/contract relief', 'E3_Fiscal measures',
              'E4_International support', 'H1_Public information campaigns', 'H2_Testing policy', 'H3_Contact tracing',
              'H4_Emergency investment in healthcare', 'H6_Facial Coverings', 'ConfirmedCases']

  lstm.calculate_shap(model, X_train, X_test, features)
  # plt.boxplot(Y_test)
  # plt.show()

  # for pos in np.flip(np.argsort(loss, axis=0))[0:10]:
  #  print(pos, X_train[pos], Y_train[pos], predictions[pos], loss[pos])

  cases_norway = data[data['CountryName'] == 'Norway']
