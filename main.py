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
  scaler = None
  for _, group in groups:
    x_group, y_group, scaler = group_to_cases(group, overlapping=overlapping)
    y = np.concatenate((y, y_group))
    x = np.concatenate((x, x_group))
  return x, y, scaler


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

  scaler = NormalizeScaler()
  scaler.fit(x)
  x, y = scaler.transform(x, y)
  return x, y, scaler


def create_supervised_data_set(data: pd.DataFrame, overlapping: bool = False):
  """
  :param data:
  :return: supervised data set (input and target)
  """
  data = data[data['ConfirmedCases'] > config.INFECTED_LOWER].groupby('CountryName')
  x, y, scaler = groups_to_cases(data, overlapping=overlapping)
  return x, y, scaler

def count_NaN_and_zero_confirmed_cases(data: pd.DataFrame):
  count_NaN = data['ConfirmedCases'].isnull().sum()
  count_zero = data['ConfirmedCases'].value_counts()[0]
  return count_NaN + count_zero

def load_and_clean_data():
  """
  :return:
  """
  data = pd.read_csv(config.DATA_PATH)

  # Remove the last two weeks (data is updated once per week, therefore the maximum gap would be two weeks)
  data = data[data['Date'] < 20201015]

  # Drop states
  data = data[pd.isnull(data['RegionName'])]

  # Keep relevant columns.
  relevant_columns = config.FEATURES + ['CountryName', 'Date', 'ConfirmedCases']
  data = data[list(filter(lambda x: x in relevant_columns, data.columns))]

  # Fill na with forward fill
  data.loc[data['Date'] == 20200101] = data.loc[data['Date'] == 20200101].fillna(0)
  if config.FFILL_ANALYSIS:
    pre_fill_NaN_zero = count_NaN_and_zero_confirmed_cases(data)
    data.fillna(method='ffill', inplace = True)
    post_fill_NaN_zero = count_NaN_and_zero_confirmed_cases(data)
    diff_fill_NaN_zero = pre_fill_NaN_zero - post_fill_NaN_zero
    total_cases_cells = len(data['ConfirmedCases'])
    # Comments made in commit id f3caa99
    print("Non NaN/zero cells forward filled:", diff_fill_NaN_zero) # 364
    print("Total cases cells:", total_cases_cells) # 53856
  else:
    data.fillna(method='ffill', inplace = True)

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

  learn_rate = [0.001, 0.005, 0.01, 0.05, 0.1]
  activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
  neurons = [10, 20, 40, 80, 160]

  model = KerasRegressor(lstm.create_model)

  params = {'learn_rate':learn_rate, 'activation':activation, 'neurons':neurons}
  clf = GridSearchCV(model, params,cv = inner_cv)
  clf.fit(x, y)

  # summarize results
  print("Best: %f using %s" % (clf.best_score_, clf.best_params_)) # average of r2 scores

  return clf.best_params_, clf

def nested_cross_validation(x: np.ndarray, y: np.ndarray):
  outer_cv = TimeSeriesSplit(n_splits=5)

  best_params, clf = grid_cross_validation(x, y)

  non_nested_r2_score = clf.best_score_
  # Nested CV with parameter optimization
  nested_r2_scores = cross_val_score(clf, X=x, y=y, cv=outer_cv)

  return best_params, non_nested_r2_score, nested_r2_scores

def visualize_predictions(model, data):
  while True:
    try:
      date_from = int(input("From:"))
      predict_days = int(input("Days:"))
      date_to = date_from + predict_days
      dates = data['Date'][date_from:date_to]
      cases_norway = data[data['CountryName'] == 'Norway']
      x, y, scaler = create_supervised_data_set(cases_norway.copy(), overlapping=True)
      X_test_norm = scaler.transform_timeseries(cases_norway[config.FEATURES].to_numpy().copy())
      prediction_norm = lstm.predict(model, X_test_norm[date_from-config.INPUTDAYS:], predict_days)
      _, prediction = scaler.inverse_transform(None, prediction_norm)
      actual = np.array(cases_norway['ConfirmedCases'][date_from:date_to])
      draw_graph({'x':dates,'y':prediction,'name':'prediction'},{'x':dates,'y':actual,'name':'actual'},
                 {'x':cases_norway['Date'][date_from-predict_days:date_from], 'y':cases_norway['ConfirmedCases'][date_from-predict_days:date_from], 'name':'start'})
    except:
      quit=input("Quit?(Y/n)")
      if quit == '' or quit == 'y' or quit == 'Y':
        return 0



def run_pipeline():
  # fix random seed for reproducibility
  seed = 10
  np.random.seed(seed)

  data = load_and_clean_data()

  x, y, _ = create_supervised_data_set(data[data['CountryName'] != 'Norway'].copy(), overlapping=True)

  if config.USE_CACHED_HYPERPARAMETERS:
    best_params = {'activation': 'tanh','learn_rate': 0.001,'neurons': 20}
  else:
    best_params, non_nested_r2_score, nested_r2_scores = nested_cross_validation(x, y) 
    # Comments made in commit: f3caa99
    print("Best params:", best_params) # Best params: {'activation': 'hard_sigmoid', 'learn_rate': 0.05, 'neurons': 20}
    print("Non-nested cross validation r2 score:", non_nested_r2_score) # Non-nested cross validation r2 score: -0.0003677288186736405
    print("Nested cross validation r2 scores:", nested_r2_scores) # Nested cross validation r2 scores: [-0.00061681 -0.00021794 -0.00018851 -0.00043963 -0.00013743]
    print("Nested cross validation r2 scores mean:", nested_r2_scores.mean()) # Nested cross validation r2 scores mean: -0.0003200653416570276

  model = lstm.create_model(**best_params)
  X_train, X_val, Y_train, Y_val = split_data(x, y)
  if config.USE_CACHED_FITTED_MODEL:
    model.load_weights('Models/model_10_0.0001.h5')
  else:
    history = lstm.train_model(model, X_train, Y_train, validation=(X_val, Y_val))
    draw_graph({'x':range(config.EPOCHS),'y':history['val_loss'],'name':'val_loss'},{'x':range(config.EPOCHS),'y':history['loss'],'name':'loss'})
    
  visualize_predictions(model, data)


  #lstm.calculate_shap(model, X_train[0:1000], X_val[1000:2000], config.FEATURES)
  # plt.boxplot(Y_test)
  # plt.show()

  # for pos in np.flip(np.argsort(loss, axis=0))[0:10]:
  #  print(pos, X_train[pos], Y_train[pos], predictions[pos], loss[pos])


if __name__ == '__main__':
  run_pipeline()
