import pandas as pd
import config
import numpy as np
import lstm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def groups_to_cases(groups, overlapping: bool = False):
  """
  :param groups:
  :param overlapping:
  :return:
  """
  y = np.array([])
  x = np.array([]).reshape(-1, config.INPUTDAYS, len(config.FEATURES))
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

  y = group['ConfirmedCases'].iloc[config.INPUTDAYS:].to_numpy()
  x = np.array([]).reshape(-1, config.INPUTDAYS, len(config.FEATURES))
  for row in range(group.shape[0] - config.INPUTDAYS):
    curr_x = group[config.FEATURES].iloc[row:row + config.INPUTDAYS].to_numpy()
    x = np.concatenate((x, curr_x.reshape(1, config.INPUTDAYS, len(config.FEATURES))), axis=0)
  return x, y


def create_supervised_data_set(data: pd.DataFrame, overlapping: bool = False):
  """
  :param data:
  :return: supervised data set (input and target)
  """
  data = data[data['ConfirmedCases'] > config.INFECTED_LOWER].groupby('CountryName')
  x, y = groups_to_cases(data, overlapping=overlapping)
  return x, y


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



def normalize_data(data: np.ndarray, scaler: MinMaxScaler):
  """
  :param data:
  :return: normalized values for x
  """
  return scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)

def de_normalize(data: np.ndarray, scaler: MinMaxScaler):
  return scaler.inverse_transform(data.reshape(-1, 1)).reshape(-1)

def test_normalize():
  scaler = MinMaxScaler()
  unnormalized = np.array([2.0,3.5,5.0])
  normalized = np.array([0.0,0.5,1.0])
  assert np.allclose(normalize_data(unnormalized, scaler), normalized)
  assert np.allclose(de_normalize(normalized, scaler), unnormalized)

if __name__ == '__main__':
  test_normalize()

  data = load_and_clean_data()
  data = data.iloc[0:200]
  # visualize_spread_for_countries(data)

  x, y = create_supervised_data_set(data[data['CountryName'] != 'Norway'], overlapping=True)
  x_norm, y_norm = x, y
  X_train, X_test, Y_train, Y_test = split_data(x_norm, y_norm)

  model = lstm.create_model()
  predictions = model.predict(X_train)
  loss = (predictions - Y_train) ** 2
  lstm.calculate_shap(model, X_train, X_test, config.FEATURES)
  # plt.boxplot(Y_test)
  # plt.show()

  # for pos in np.flip(np.argsort(loss, axis=0))[0:10]:
  #  print(pos, X_train[pos], Y_train[pos], predictions[pos], loss[pos])

  cases_norway = data[data['CountryName'] == 'Norway']
