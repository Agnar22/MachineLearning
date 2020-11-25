import pandas as pd
import config
import numpy as np
import lstm
import visualization
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from dataprocessing import difference, undo_difference
from visualization import draw_graph
from scaler import NormalizeScaler

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

  data = data[:2000]

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

def split_data(x: np.ndarray, y: np.ndarray):
  """
  :param x:
  :param y:
  :return:
  """
  X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=config.VALIDATION_SIZE, shuffle=False)
  return X_train, X_test, Y_train, Y_test

def grid_cross_validation(x: np.ndarray, y: np.ndarray):
  inner_cv = TimeSeriesSplit(n_splits=5)

  learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
  activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
  dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  neurons = [32, 64, 128, 256, 512]

  pipeline = make_pipeline(NormalizeScaler(),KerasRegressor(lstm.create_model))

  params = {'kerasregressor__learn_rate':learn_rate, 'kerasregressor__activation':activation, 'kerasregressor__dropout_rate':dropout_rate, 'kerasregressor__neurons':neurons}
  clf = GridSearchCV(pipeline, params,cv = inner_cv)
  grid_result = clf.fit(x, y)

  # summarize results
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) # average of r2 scores

  return grid_result.best_params_, clf

def nested_cross_validation(x: np.ndarray, y: np.ndarray):
  outer_cv = TimeSeriesSplit(n_splits=5)

  best_params, clf = grid_cross_validation(x, y)

  # Nested CV with parameter optimization
  r2_scores = cross_val_score(clf, X=x, y=y, cv=outer_cv)

  return best_params, r2_scores
  

def run_pipeline():
  data = load_and_clean_data()

  x, y = create_supervised_data_set(data[data['CountryName'] != 'Norway'], overlapping=True)

  #best_params = {'learn_rate': 0.001,'activation': 'relu', 'dropout_rate': 0.2, 'neurons': 128}
  best_params, _ = grid_cross_validation(x, y)
  #best_params, r2_scores = nested_cross_validation(x, y)
  #print("Nested cross validation r2 scores:" + r2_scores)
  #print("Nested cross validation r2 scores mean:" + r2_scores.mean())

  model = lstm.create_model(**best_params)
  X_train, X_test, Y_train, Y_test = split_data(x, y)
  scaler = NormalizeScaler()
  X_train_norm, X_test_norm, Y_train_norm, Y_test_norm = scaler.normalize_train_test(X_train, X_test, Y_train, Y_test)
  lstm.train_model(model, X_train_norm, Y_train_norm, validation=(X_test_norm, Y_test_norm))
  date_from = 0#250
  predict_days = 30
  date_to = date_from + predict_days
  dates = data['Date'][date_from:date_to]
  #prediction_norm = lstm.predict(model, data[config.FEATURES].to_numpy()[date_from:,:], predict_days)
  #prediction = de_normalize(prediction_norm, scalers[-1])
  #actual = data['ConfirmedCases'][date_from:date_to]

  #draw_graph({'x':dates,'y':prediction,'name':'prediction'},{'x':dates,'y':actual,'name':'actual'})

  return

  lstm.calculate_shap(model, x_norm[0:1000], x_norm[1000:2000], config.FEATURES)
  # plt.boxplot(Y_test)
  # plt.show()

  # for pos in np.flip(np.argsort(loss, axis=0))[0:10]:
  #  print(pos, X_train[pos], Y_train[pos], predictions[pos], loss[pos])

  cases_norway = data[data['CountryName'] == 'Norway']

if __name__ == '__main__':
  run_pipeline()
