from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
class ARIMA():
  def __init__(self, x, p, d, q):
    x = self.difference(x, d) # make x stationary

    # generate random phi values following the constraints: -1 < phi < 1, sum(phi) < 1, etc..
    phi = self.generate_phi_random(p)
    # generate random theta values following the constrains: -1 < theta < 1,  sum(theta) > -1, etc..
    theta = self.generate_theta_random(q)

    c = self.mean(x)*(1-sum(phi))  # https://people.duke.edu/~rnau/arimest.html

    e_temp = self.AR_process(c, p, phi, x)  # errors calculated by running AR
    e = self.MA_process(q, theta, e_temp)  # errors calculated by running AR+MA

    self.x = x
    self.p = p
    self.d = d
    self.q = q
    self.phi = phi
    self.theta = theta
    self.c = c
    self.e = e

  def forecast(self, h):
    n = len(self.x)
    for t in range(n, n+h):
      sum_phi_x = 0
      for i in range(self.p):
        sum_phi_x += self.phi[i] * self.x[t-(i+1)]
      sum_theta_e = 0
      for i in range(self.q):
        sum_theta_e += self.theta[i] * self.e[t-(i+1)]
      x_new = self.c + sum_phi_x + sum_theta_e
      self.x = np.append(self.x, x_new)
      self.e = np.append(self.e, 0)
    forecast = self.undo_difference(self.x, self.d)[-h-1:]
    return forecast

  def AR_process(self, c, p, phi, x):
    n = len(x)
    e = [0 for i in range(n)]
    for t in range(p, n):
      sum_phi_x = 0
      for i in range(p):
        sum_phi_x += phi[i] * x[t-(i+1)]
      e[t] = x[t] - (c + sum_phi_x)
    return e

  def MA_process(self, q, theta, e):
    n = len(e)
    for t in range(q, n):
      sum_theta_e = 0
      for i in range(q):
        sum_theta_e += theta[i] * e[t-(i+1)]
      e[t] -= sum_theta_e
    return e

  def generate_phi_random(self, p):
    return [0.2,0,0,0,0,0,0,0,0,0,0, 0.8] 

  def generate_theta_random(self, q):
    return [0.4, 0.4]

  def mean(self, x):
    return sum(x)/len(x)

  def difference(self, x, d=1):
      for _ in range(d):
        x = np.diff(x, prepend=0) # out[i] = x[i] - x[i-1], out[0]=x[0]
      return x

  def undo_difference(self, x, d=1):
      for _ in range(d):
        x = np.cumsum(x) # out[i] = x[i] + out[i-1], out[0]=x[0]
      return x

if __name__ == "__main__":
  rcParams['figure.figsize'] = 10, 6
  path = "C:/Users/eivin/OneDrive/Skrivebord/AirPassengers.csv" 
  dataset = pd.read_csv(path)["#Passengers"].to_numpy()
  arima = ARIMA(dataset,12,1,2)
  forecast = arima.forecast(len(dataset) // 5)

  #plt.plot(forecast, "r")
  plt.plot(dataset, "r")
  x_shifted = [len(dataset) - 1 + i for i in range(len(forecast))]
  plt.plot(x_shifted, forecast, "b")
  plt.show()
