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
  model = pm.auto_arima(dataset, maxiter = 100, seasonal=True, m = 7)
  print(model.summary())
  forecast = model.predict(50)
  plt.plot(dataset, "r")
  x_shifted = [i + (len(dataset)) for i in range(50)]
  plt.plot(x_shifted, forecast, "b")
  plt.show()
