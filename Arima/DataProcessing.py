import numpy as np
import statsmodels
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def is_stationary(series):
  adfTest = adfuller(series, autolag='AIC')
  SignificanceLevel = .05
  pValue = adfTest[1]
  return pValue<SignificanceLevel

def difference(series, d=1):
  for _ in range(d):
    series = np.diff(series, prepend=0) # out[i] = y[i] - y[i-1], out[0]=y[0]
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
  return (series, d)