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

predict_num = 30

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
  methods = ['newton','nm', 'bfgs', 'lbfgs', 'powell', 'cg']
  for method_arg in methods:
    print(method_arg)

    #d trend
    #P mellom 1 og 2 pga spikes
    #D seasonality
    #Q no spikes
    #m seasonality lag
    model = pm.arima.AutoARIMA(d = 1, seasonal=True, D = 1, m = 7, suppress_warnings=True)

    #resid = model.resid()
    #print(normaltest(resid))
    #plt.hist(resid)
    #plt.show()
    #model.fit()
    forecast = model.fit_predict(y = dataset[:-predict_num], n_periods = predict_num)
    print(model.summary())
    plt.plot(dataset, "r")
    x_shifted = [i + (len(dataset[:-predict_num])) for i in range(predict_num)]
    plt.plot(x_shifted, forecast, "b")
    plt.show()
