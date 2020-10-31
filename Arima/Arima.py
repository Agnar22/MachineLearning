from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
import DataProcessing as dp
from matplotlib.pylab import rcParams
import os
class ARIMA():
  def __init__(self, series):
    #TODO: PACF and ACD
    self.p = 12
    self.q = 2

    _series, _d = dp.make_stationary(series)
    self.series = _series
    self.d = _d

    #TODO: MLE for phi and theta
    self.phi = self.generate_example_phi(self.p)
    self.theta = self.generate_example_theta(self.q)

    self.c = self.mean(self.series)*(1-sum(self.phi))  # https://people.duke.edu/~rnau/arimest.html

    e_temp = self.AR_process(self.c, self.p, self.phi, self.series)  # errors calculated by running AR
    self.e = self.MA_process(self.q, self.theta, e_temp)  # errors calculated by running AR+MA

  def forecast(self, h):
    n = len(self.series)
    for t in range(n, n+h):
      sum_phi_x = 0
      for i in range(self.p):
        sum_phi_x += self.phi[i] * self.series[t-(i+1)]
      sum_theta_e = 0
      for i in range(self.q):
        sum_theta_e += self.theta[i] * self.e[t-(i+1)]
      x_new = self.c + sum_phi_x + sum_theta_e
      self.series = np.append(self.series, x_new)
      self.e = np.append(self.e, 0)
    forecast = dp.undo_difference(self.series, self.d)[-h-1:]
    return forecast

  def AR_process(self, c, p, phi, series):
    n = len(series)
    e = [0 for i in range(n)]
    for t in range(p, n):
      sum_phi_series = 0
      for i in range(p):
        sum_phi_series += phi[i] * series[t-(i+1)]
      e[t] = series[t] - (c + sum_phi_series)
    return e

  def MA_process(self, q, theta, e):
    n = len(e)
    for t in range(q, n):
      sum_theta_e = 0
      for i in range(q):
        sum_theta_e += theta[i] * e[t-(i+1)]
      e[t] -= sum_theta_e
    return e

  def generate_example_phi(self, p):
    return [0.2,0,0,0,0,0,0,0,0,0,0, 0.8] 

  def generate_example_theta(self, q):
    return [0.4, 0.4]

  def mean(self, series):
    return sum(series)/len(series)

if __name__ == "__main__":
  rcParams['figure.figsize'] = 10, 6
  path = os.path.dirname(__file__) + "\\AirPassengers.csv" 
  dataset = pd.read_csv(path)["#Passengers"].to_numpy()
  arima = ARIMA(dataset)
  forecast = arima.forecast(len(dataset) // 5)

  plt.plot(dataset, "r")
  x_shifted = [len(dataset) - 1 + i for i in range(len(forecast))]
  plt.plot(x_shifted, forecast, "b")
  plt.show()
