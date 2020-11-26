import numpy as np
import statsmodels
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def is_stationary(series):
  adfTest = adfuller(series, autolag='AIC')
  SignificanceLevel = .05
  pValue = adfTest[1]
  return pValue<SignificanceLevel

def difference(x, d=1):
  #for _ in range(d):
  #  x = np.diff(x, prepend=0) # out[i] = x[i] - x[i-1], out[0]=x[0]
  return x

def undo_difference(x, d=1):
  #for _ in range(d):
  #  x = np.cumsum(x) # out[i] = x[i] + out[i-1], out[0]=x[0]
  return x

def make_stationary(series):
  d = 0
  while not is_stationary(series):
    series = difference(series)
    d += 1
    if d == 100:
      raise Exception("Could not differentiate the data")
  return (series, d)