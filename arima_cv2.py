from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
import pmdarima as pm
import os
import config
from main import create_supervised_data_set, load_and_clean_data
from scipy import stats
from scipy.stats import normaltest
from pmdarima import model_selection

def get_cases(data: pd.DataFrame):
  return data[data['location'] == config.COUNTRIES[0]]['total_cases_per_million'].values

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def cross_validation_parameters(dataset):
  p_values = range(0, 7)
  q_values = range(0, 7)
  P_values = range(0, 3)
  Q_values = range(0, 3)

  p_values = range(1,3)
  q_values = range(1,2)
  P_values = range(1,2)
  Q_values = range(1,2)

  best_score, best_parameters = float("inf"), None
  cv = model_selection.RollingForecastCV(step=10, h=30)
  for p in p_values:
    for q in q_values:
      for P in P_values:
        for Q in Q_values:
          model = pm.arima.ARIMA(order=(p, 1, q), seasonal_order=(P, 1, Q, 7), suppress_warnings=True) 
          cv_scores = model_selection.cross_val_score(model, dataset, scoring=mape,cv=cv, verbose=2) #, scoring='smape'
          score = np.average(cv_scores)
          parameters = (p,q,P,Q)
          print(p,q,P,Q,"- mape:",score)
          if score < best_score:
            best_score, best_parameters = score, parameters
  print("Best parameters:", best_parameters)
  return best_parameters
# nested cross validation brukes for error estimerings delen
# prøv å få hyper parameters og prøv å få score for rapport

def run():
  data = get_cases(load_and_clean_data())[:-30]

  train, test = model_selection.train_test_split(data, test_size=30)

  p,q,P,Q = cross_validation_parameters(train)

  model = pm.arima.ARIMA(order=(p, 1, q), seasonal_order=(P, 1, Q, 7), suppress_warnings=True)
  forecast = model.fit_predict(y = train, n_periods = 30)

  plt.plot(data, "r")
  x_shifted = [i + len(train) for i in range(len(test))]
  plt.plot(x_shifted, forecast, "b")
  print("Final mape:", mape(test, forecast))
  plt.show()

if __name__ == "__main__":
  run()