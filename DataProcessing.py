import numpy as np
import statsmodels
from statsmodels.tsa.stattools import adfuller
import pandas as pd

class DataProcessing():
  def __init__(self):
    self.SignificanceLevel = .05
    self.pValue = None
    self.isStationary = None

  def is_stationary(self, series):
    adfTest = adfuller(series, autolag='AIC')
    self.pValue = adfTest[1]
    return self.pValue<self.SignificanceLevel

  def difference(self, series, d=1):
    for _ in range(d):
      series = np.diff(series, prepend=0) # out[i] = y[i] - y[i-1], out[0]=y[0]
    return series

  def undo_difference(self, series, d=1):
      for _ in range(d):
        series = np.cumsum(series) # out[i] = y[i] + out[i-1], out[0]=y[0]
      return series

  def make_stationary(self, series):
    d = 0
    while not self.is_stationary(series):
      series = self.difference(series)
      d += 1
    return (series, d)