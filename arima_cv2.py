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

# nested cross validation brukes for error estimerings delen
# prøv å få hyper parameters og prøv å få score for rapport

def run():
  y = get_cases(load_and_clean_data())

  #est = pm.arima.ARIMA(order=(5, 3, 0), seasonal_order=(0, 0, 1, 7), suppress_warnings=True) 
  est = pm.arima.ARIMA(order=(5, 3, 0), seasonal_order=(0, 0, 1, 7), suppress_warnings=True) 
  est.fit(y)
  print(normaltest(est.resid()))
  cv = model_selection.RollingForecastCV(step=5, h=30, initial=90)
  predictions = model_selection.cross_val_predict(
      est, y, cv=cv, verbose=2)
  score = model_selection.cross_val_score(
      est, y, scoring = 'mean_squared_error', cv=cv, verbose=2)
  print("Model 1 CV scores: {}".format(score.tolist()))

  # plot the predictions over the original series
  x_axis = np.arange(y.shape[0])
  n_test = predictions.shape[0]

  plt.plot(x_axis, y, alpha=0.75, c='b')
  plt.plot(x_axis[-n_test:], predictions, alpha=0.75, c='g')  # Forecasts
  plt.title("Cross-validated forecasts")
  plt.show()


if __name__ == "__main__":
  run()