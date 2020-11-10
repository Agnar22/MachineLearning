import numpy as np
import statsmodels
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def is_stationary(series):
  adfTest = adfuller(series, autolag='AIC')
  SignificanceLevel = .05
  pValue = adfTest[1]
  return pValue<SignificanceLevel

def difference(series, d=1, lag=1):
  for _ in range(d):
    differenced_series = []
    series = np.insert(series,0,0,axis=0) # prepend 0 to keep starting value
    for n in range(len(series) - lag):
      differenced_series.append(series[n+lag] - series[n]) # out[i] = y[i] - y[i-1], out[0]=y[0]
    series = np.array(differenced_series)
  return series

def undo_difference(series, d=1):
    for _ in range(d):
      series = np.cumsum(series) # out[i] = y[i] + out[i-1], out[0]=y[0]
    return series

def make_stationary(series):
  d = 0
  while not is_stationary(series):
    series = difference(series)
    d += 1
    if d == 100:
      raise Exception("Could not differentiate the data")
  return (series, d)

if __name__ == "__main__":
  print(difference([0,1,2], d=1, lag=2))

  #Finne riktig p,q,d, så jeg ikke trenger å tenke på det shittet (pacf,acf/grid)
  #Teste hvilken method som er best