from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
import pmdarima as pm
import os
import config
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from main import create_supervised_data_set, load_and_clean_data
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest
from dataprocessing import make_stationary, difference
import statsmodels.api as sm

def visualize_data(data: pd.DataFrame):
  # clearly a seasonal pattern, m = 7
  plot_acf(data)
  plt.show()

  # 
  plot_pacf(data)
  plt.show()


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
    if isinstance(exog_data[0], float):
      toReturn.append(exog_data[:-90])
  #print(toReturn)
  return np.column_stack(toReturn)
  #return np.array(toReturn)

def load_and_clean_exog_data2():
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
    if isinstance(exog_data[0], float):
      #print(exog_data[:-90])
      #model = pm.arima.AutoARIMA(maxiter = 100, max_d = 3)
      #toReturn.append(model.fit_predict(y = exog_data[:-90], n_periods = 87))
      toReturn.append(exog_data[-90:-3])
  #print(toReturn)
  return np.column_stack(toReturn)
  #return np.array(toReturn)

def check_residual(data :pd.DataFrame):
  #arima_mod6 = sm.tsa.ARIMA(data, (6,0,0)).fit(disp=False) # disp = print output

  arima_mod6 = pm.arima.ARIMA(order=(6, 1, 0), seasonal_order=(0, 0, 1, 7), maxiter = 100) 
  arima_mod6.fit(data[1:-90])
  print(arima_mod6.arparams())
  # arima_mod6.plot_diagnostics()
  # plt.show()
  # returns a 2-tuple of the chi-squared statistic, and the associated p-value. the p-value is very small, meaning
  # the residual is not a normal distribution
  resid = arima_mod6.resid()
  print(normaltest(resid))

  print(np.mean(resid))
  #plt.hist(resid)
  #plt.show()

  forecast = arima_mod6.predict(87)
  plt.plot(data[:-3], "r")
  x_shifted = [i + (len(data[:-90])) for i in range(87)]
  plt.plot(x_shifted, forecast, "b")
  plt.show()

  #visualize_data(resid)

def get_cases(data: pd.DataFrame):
  return data[data['location'] == config.COUNTRIES[0]]['total_cases_per_million'].values

if __name__ == "__main__":
  dataset = get_cases(load_and_clean_data())  

  dataset_stationary = difference(dataset, d = 1, lag = 1)
  dataset_stationary = difference(dataset_stationary, d = 1, lag = 7)
  dataset_stationary, d = make_stationary(dataset_stationary)
  print(d)
  #dataset_stationary = difference(dataset, d = 1, lag = 1)
  visualize_data(dataset_stationary)

  #check_residual(dataset)

  


  # Get the fitted parameters used by the function
  
  # tscv = TimeSeriesSplit(n_splits = 4)
  # rmse = []

  # for train_index, test_index in tscv.split(dataset):
  #     cv_train, cv_test = dataset.iloc[train_index], dataset.iloc[test_index]
      
  #     pmd
      
  #     predictions = arma.predict(cv_test.index.values[0], cv_test.index.values[-1])
  #     true_values = cv_test.values
  #     rmse.append(sqrt(mean_squared_error(true_values, predictions)))

  # model = pm.auto_arima(dataset[:-90], maxiter = 100, d = 3)
  # pm.auto_arima.fit()
  # print(model.summary())
  # forecast = model.predict(90)
  # plt.plot(dataset, "r")
  # x_shifted = [i + (len(dataset[:-90])) for i in range(90)]
  # plt.plot(x_shifted, forecast, "b")
  # plt.show()
    
#print("RMSE: {}".format(np.mean(rmse)))