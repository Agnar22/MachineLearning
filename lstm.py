from keras.layers import Dense, LSTM, Dropout, Conv1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from typing import List
import matplotlib.pyplot as plt
#import shap
import numpy as np
import pandas as pd
import config
import main


def create_model(learn_rate, activation, dropout_rate, neurons):
  print(learn_rate, activation, dropout_rate, neurons)
  model = Sequential()
  model.add(
    LSTM(config.UNITS, input_shape=(config.INPUTDAYS, len(config.FEATURES)), return_sequences=False))
  model.add(Dropout(dropout_rate))
  model.add(Dense(neurons*2, activation=activation))
  model.add(Dropout(dropout_rate))
  model.add(Dense(neurons*2, activation=activation))
  model.add(Dropout(dropout_rate))
  model.add(Dense(neurons, activation=activation))
  model.add(Dropout(dropout_rate))
  model.add(Dense(neurons, activation=activation))
  model.add(Dropout(dropout_rate))
  model.add(Dense(neurons, activation=activation))
  model.add(Dropout(dropout_rate))
  model.add(Dense(neurons, activation=activation))
  # model.add(Dropout(dropout_rate))
  model.add(Dense(neurons, activation=activation))
  model.add(Dense(1, activation='linear'))
  optim = Adam(lr=0.0001)
  # optim = RMSprop()
  model.compile(loss='mse', optimizer=optim)
  model.summary()
  return model


def train_model(model: Sequential, X_train: np.ndarray, Y_train: np.ndarray, validation=None):
  if config.SAVE:
    model_checkpoint_callback = ModelCheckpoint(
      filepath='Models/model_{epoch}_{val_loss:.4f}.h5',
      save_weights_only=False,
      monitor='val_loss',
      mode='auto',
      save_best_only=True
    )

    train_history = model.fit(
      X_train,
      Y_train,
      batch_size=config.BATCH_SIZE,
      epochs=config.EPOCHS,
      validation_data=validation,
      callbacks=[model_checkpoint_callback]
    )
  else:
    train_history = model.fit(X_train, Y_train, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, validation_data=validation)
  return pd.DataFrame.from_dict(train_history.history)




def predict(model: Sequential, x: np.ndarray, days: int, series_dim: int = -1):
  predictions = np.array([])
  rec_x = x.copy()

  # Make recursive predictions.
  for day in range(days):
    inp = rec_x[day:day + config.INPUTDAYS].reshape(1, config.INPUTDAYS, len(config.FEATURES))
    pred = model.predict(inp) * 0.001
    predictions = np.append(predictions, pred)
    rec_x[day + config.INPUTDAYS, series_dim] = pred
    # rec_x = np.append(rec_x[1:], pred)
  return predictions
