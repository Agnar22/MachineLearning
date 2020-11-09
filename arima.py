from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
import pmdarima as pm
import os
import config
from pmdarima import model_selection
from main import create_supervised_data_set, load_and_clean_data

def get_cases(data: pd.DataFrame):
  return data[data['location'] == config.COUNTRIES[0]]['total_cases_per_million'].values

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def run():
  data = get_cases(load_and_clean_data())
  train, test = model_selection.train_test_split(data, test_size=30)

  # Even though we have a dedicated train/test split, we can (and should) still
  # use cross-validation on our training set to get a good estimate of the model
  # performance. We can choose which model is better based on how it performs
  # over various folds.
  model1 = pm.ARIMA(order=(6, 3, 0),
                    seasonal_order=(0, 0, 1, 7),
                    suppress_warnings=True)
  model2 = pm.ARIMA(order=(5, 3, 0),
                    seasonal_order=(0, 0, 1, 7),
                    suppress_warnings=True,)
  cv = model_selection.SlidingWindowForecastCV(step=1, h=30)

  model1_cv_scores = model_selection.cross_val_score(
      model1, train, scoring='smape', cv=cv, verbose=2)

  model2_cv_scores = model_selection.cross_val_score(
      model2, train, scoring='smape', cv=cv, verbose=2)

  print("Model 1 CV scores: {}".format(model1_cv_scores.tolist()))
  print("Model 2 CV scores: {}".format(model2_cv_scores.tolist()))

  # Pick based on which has a lower mean error rate
  m1_average_error = np.average(model1_cv_scores)
  m2_average_error = np.average(model2_cv_scores)
  errors = [m1_average_error, m2_average_error]
  models = [model1, model2]

  # print out the answer
  better_index = np.argmin(errors)  # type: int
  print("Lowest average SMAPE: {} (model{})".format(
      errors[better_index], better_index + 1))
  print("Best model: {}".format(models[better_index]))

def old():
  dataset = get_cases(load_and_clean_data())
  predict_amount = 30
  model = pm.arima.ARIMA(order=(6, 3, 0), seasonal_order=(0, 0, 1, 7), maxiter = 100) 
  model.fit(dataset[:-predict_amount])
  forecast = model.predict(predict_amount)
  #model = pm.arima.AutoARIMA(maxiter = 100, d = 3, seasonal=True, m = 7)
  #forecast = model.fit_predict(y = dataset[:-30], n_periods = 30)
  print(model.summary())
  
  plt.plot(dataset, "r")
  x_shifted = [i + (len(dataset[:-predict_amount])) for i in range(predict_amount)]
  plt.plot(x_shifted, forecast, "b")
  plt.show()

  print(mape(dataset[-predict_amount:], forecast))

if __name__ == "__main__":
  run()

