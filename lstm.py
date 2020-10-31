from keras.layers import Dense, Flatten, LSTM, RepeatVector
from keras.models import Sequential
import numpy as np
import config


def create_model():
  model = Sequential()
  model.add(LSTM(config.UNITS, input_shape=(config.INPUTDAYS, 1), return_sequences=False))
  model.add(Dense(1))
  model.compile(loss='mse', optimizer='adam')
  model.summary()
  return model


def train_model(model: Sequential, X_train: np.ndarray, Y_train: np.ndarray, validation=None):
  print(X_train.shape)
  print(Y_train.shape)
  Y_train = Y_train.reshape(*Y_train.shape)
  print(model.predict(X_train))
  model.fit(x=X_train[0:300], y=Y_train[0:300], epochs=50, batch_size=50, verbose=True)
