from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
import pmdarima as pm
import os
import config
from main import create_supervised_data_set, load_and_clean_data

def get_cases(data: pd.DataFrame):
  return data[data['location'] == config.COUNTRIES[0]]['total_cases_per_million'].values

if __name__ == "__main__":
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
