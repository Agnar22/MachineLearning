import pandas as pd
import config
import numpy as np
import lstm
import visualization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from dataprocessing import difference, undo_difference
from visualization import draw_graph


def groups_to_cases(groups, overlapping: bool = False):
  """
  :param groups:
  :param overlapping:
  :return:
  """
  y = np.array([])
  x = np.array([]).reshape(-1, config.INPUTDAYS, len(config.FEATURES))
  for _, group in groups:
    #group["ConfirmedCases"] = difference(group["ConfirmedCases"], 2)
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

def count_NaN_and_zero_confirmed_cases(data: pd.DataFrame):
  count_NaN = data['ConfirmedCases'].isnull().sum()
  count_zero = data['ConfirmedCases'].value_counts()[0]
  return count_NaN + count_zero

def load_and_clean_data():
  """
  :return:
  """
  data = pd.read_csv(config.DATA_PATH)

  data = data.iloc[:200]

  # Remove the last two weeks (data is updated once per week, therefore the maximum gap would be two weeks)
  data = data[data['Date'] < 20201015]

  # Drop states
  data = data[pd.isnull(data['RegionName'])]

  # Keep relevant columns.
  relevant_columns = config.FEATURES + ['CountryName', 'Date', 'ConfirmedCases']
  data = data[list(filter(lambda x: x in relevant_columns, data.columns))]

  # Fill na with forward fill
  data.loc[data['Date'] == 20200101] = data.loc[data['Date'] == 20200101].fillna(0)
  data.fillna(method='ffill', inplace = True)

  # Forward fill analysis:
  # count_NaN_and_zero_confirmed_cases(data) was equal to 13677 before fillna, and 13313 after fillna.
  # Meaning 364 cells with value higher than zero confirmed cases were forward filled.
  # len(data['ConfirmedCases']) was equal to 53856
  # Tested in commit id e8d334

  # Format date
  data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')

  return data


def normalize_dataset(X,Y):
  scalers = []
  for col in range(X.shape[2]):
    scaler = MinMaxScaler()
    X[:,:,col] = normalize_data(X[:, :, col], scaler).reshape(*X.shape[:2])
    scalers.append(scaler)
  Y = normalize_data(Y, scaler) if Y is not None else None  # using the last scaler as it is the confirmed case scaler
  return X, Y, scalers

def normalize_data(data: np.ndarray, scaler: MinMaxScaler):
  """
  :param data:
  :return: normalized values for x
  """
  return scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)

def de_normalize(data: np.ndarray, scaler: MinMaxScaler):
  return scaler.inverse_transform(data.reshape(-1, 1)).reshape(-1)

def cross_validation(x, y):
  inner_cv = TimeSeriesSplit(n_splits=5)

  lstm_model = KerasClassifier(build_fn=lstm.create_model, verbose=2)

  learn_rate = [0.001]#[0.001, 0.01, 0.1, 0.2, 0.3]
  activation = ['relu']#['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
  dropout_rate = [0.2]#[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

  params = dict(learn_rate=learn_rate, activation=activation, dropout_rate=dropout_rate)
  clf = GridSearchCV(estimator=lstm_model, param_grid=params, cv = inner_cv)
  grid_result = clf.fit(x, y)

  # summarize results
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) # average of r2 scores

  return grid_result.best_estimator_, clf

def nested_cross_validation(x, y):
  outer_cv = TimeSeriesSplit(n_splits=5)

  _, clf = cross_validation(x, y)
  non_nested_score = clf.best_score_

  # Nested CV with parameter optimization
  nested_score = cross_val_score(clf, X=x, y=y, cv=outer_cv)
  nested_score = nested_score.mean()

  score_difference = non_nested_score - nested_score

  print("CV score:", non_nested_score)
  print("NCV score:", nested_score)
  print("Difference of", score_difference)
  

def run_pipeline():
  data = load_and_clean_data()

  x, y = create_supervised_data_set(data[data['CountryName'] != 'Norway'], overlapping=True)

  x_norm, y_norm, scalers = normalize_dataset(x.copy(), y.copy())

  #nested_cross_validation(x_norm, y_norm)
  #best_params = cross_validation(x_norm, y_norm)
  #best_params = {'activation': 'relu', 'dropout_rate': 0.2, 'learn_rate': 0.001}

  model,_ = cross_validation(x_norm, y_norm) # lstm.create_model(**best_params) 
  model = lstm.create_model(0.2, 'relu', 0.25)
  date_from = 20
  predict_days = 30
  date_to = date_from + predict_days
  dates = data['Date'][date_from:date_to]
  prediction_norm = lstm.predict(model, data[config.FEATURES].to_numpy()[date_from:,:], predict_days)
  prediction = de_normalize(prediction_norm, scalers[-1])
  actual = data['ConfirmedCases'][date_from:date_to]

  draw_graph({'x':dates,'y':prediction,'name':'prediction'},{'x':dates,'y':actual,'name':'actual'})

  return

  lstm.calculate_shap(model, x_norm[0:1000], x_norm[1000:2000], config.FEATURES)
  # plt.boxplot(Y_test)
  # plt.show()

  # for pos in np.flip(np.argsort(loss, axis=0))[0:10]:
  #  print(pos, X_train[pos], Y_train[pos], predictions[pos], loss[pos])

  cases_norway = data[data['CountryName'] == 'Norway']

if __name__ == '__main__':
  run_pipeline()
