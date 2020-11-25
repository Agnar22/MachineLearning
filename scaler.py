from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import config
class NormalizeScaler( BaseEstimator, TransformerMixin ):
  def __init__( self):
    self.scalers = []

  def normalize_data(self, data: np.ndarray, scaler: MinMaxScaler):
    """
    :param data:
    :return: normalized values for x
    """
    return scaler.transform(data.reshape(-1, 1)).reshape(-1)

  def de_normalize_data(self, data: np.ndarray, scaler: MinMaxScaler):
    """
    :param data:
    :return: de_normalized values for x
    """
    return scaler.inverse_transform(data.reshape(-1, 1)).reshape(-1)

  def fit( self, X, y = None ):
    self.scalers = []
    for col in range(X.shape[2]):
      scaler = MinMaxScaler()
      scaler.fit(X[:, :, col].reshape(-1,1))
      self.scalers.append(scaler)
    return self
  
  def transform( self, X, y = None ):
    for col in range(X.shape[2]):
      scaler = self.scalers[col]
      X[:,:,col] = self.normalize_data(X[:, :, col], scaler).reshape(*X.shape[:2])
    cases_index = config.FEATURES.index('ConfirmedCases')
    if y is None:
      return X
    y = self.normalize_data(y, self.scalers[cases_index])
    return X, y

  def inverse_transform(self, X, y=None):
    for col in range(X.shape[2]):
      scaler = self.scalers[col]
      X[:,:,col] = self.de_normalize_data(X[:, :, col], scaler).reshape(*X.shape[:2])
    cases_index = config.FEATURES.index('ConfirmedCases')
    if y is None:
      return X
    y = self.de_normalize_data(y, self.scalers[cases_index])
    return X, y

  def normalize_train_test(self, X_train, X_test, y_train, y_test):
    self.fit(X_train, y_train)
    X_train, y_train = self.transform(X_train, y_train)
    X_test, y_test = self.transform(X_test, y_test)
    return X_train, X_test, y_train, y_test