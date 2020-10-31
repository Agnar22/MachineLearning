from keras.layers import Dense, Flatten, LSTM, RepeatVector
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import config


def create_model():
  model = Sequential()
  model.add(LSTM(config.UNITS, input_shape=(config.INPUTDAYS, 1), return_sequences=False))
  model.add(Dense(1))
  optim = Adam(learning_rate=0.0001)
  model.compile(loss='mse', optimizer=optim)
  model.summary()
  return model


def train_model(model: Sequential, X_train: np.ndarray, Y_train: np.ndarray, validation=None):
  model.fit(x=X_train[0:300], y=Y_train[0:300], epochs=50, batch_size=50, verbose=True)
