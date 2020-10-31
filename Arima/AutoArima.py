from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
import pmdarima as pm
import os

if __name__ == "__main__":
  rcParams['figure.figsize'] = 10, 6
  path = os.path.dirname(__file__) + "\\AirPassengers.csv" 
  dataset = pd.read_csv(path)["#Passengers"].to_numpy()

  # TODO: Find p and q, since I think that is done pretty badly in auto_arima
  # TODO: Find out how this model trains (standard approuch is nested cross-validation/MLE)
  model = pm.auto_arima(dataset, seasonal=True, maxiter = 100, m=12)
  print(model.summary())
  forecast = model.predict(50)
  plt.plot(dataset, "r")
  x_shifted = [i + (len(dataset)) for i in range(50)]
  plt.plot(x_shifted, forecast, "b")
  plt.show()
