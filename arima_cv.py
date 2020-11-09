from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
import pmdarima as pm
import os
import config
from main import create_supervised_data_set, load_and_clean_data
from pmdarima.model_selection import RollingForecastCV
from pmdarima import model_selection

def get_cases(data: pd.DataFrame):
  return data[data['location'] == config.COUNTRIES[0]]['total_cases_per_million'].values

def foo_loss(y_true, y_pred):
  return abs((y_true-y_pred)/y_true)

if __name__ == "__main__":
  dataset = get_cases(load_and_clean_data())
  model = pm.ARIMA(order=(5, 3, 0), out_of_sample_size=30, scoring = foo_loss) 
  cv = RollingForecastCV()
  model1_cv_scores = model_selection.cross_val_score(
  model, dataset[:-90], scoring='smape', cv=cv, verbose=2)
  print("Model 1 CV scores: {}".format(model1_cv_scores.tolist()))

  # forecast = model.predict(90)
  # plt.plot(dataset, "r")
  # x_shifted = [i + (len(dataset[:-90])) for i in range(90)]
  # plt.plot(x_shifted, forecast, "b")
  # plt.show()
