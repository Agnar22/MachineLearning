from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
import pmdarima as pm
import os
import config
from main import create_supervised_data_set, load_and_clean_data
from scipy.stats import normaltest

def get_cases(data: pd.DataFrame):
  return data[data['location'] == config.COUNTRIES[0]]['total_cases_per_million'].values

def load_and_clean_exog_data():
  """
  :return:
  """
  data = pd.read_csv('Data/OxCGRT_latest.csv')
  newDate = []
  for n in range(len(data['Date'])):
    d = str(data['Date'][n])
    newDate.append(d[0:4] + "-" + d[4:6] + "-" + d[6:8])
  data['Date'] = newDate
  data['Date'] = pd.DatetimeIndex(data['Date'])
  data = data.fillna(0)
  toReturn = []
  for key in data[data['CountryName'] == config.COUNTRIES[0]]:
    exog_data = np.array(data[data['CountryName'] == config.COUNTRIES[0]][key].values)
    if not isinstance(exog_data[0], str):
      toReturn.append(exog_data[1:-1])
  #print(toReturn)
  #return np.column_stack(toReturn)
  return np.array(toReturn)

if __name__ == "__main__":
  dataset = get_cases(load_and_clean_data())
  model = pm.arima.AutoARIMA(maxiter = 100, d = 3, seasonal=True, m = 7)

  #resid = model.resid()
  #print(normaltest(resid))
  #plt.hist(resid)
  #plt.show()
  #print(model.summary())
  forecast = model.fit_predict(y = dataset[:-90], exogenous = load_and_clean_exog_data(), n_periods = 90)
  plt.plot(dataset, "r")
  x_shifted = [i + (len(dataset[:-90])) for i in range(90)]
  plt.plot(x_shifted, forecast, "b")
  plt.show()
