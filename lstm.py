from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from typing import List
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
import config
import main


def create_model(learn_rate, activation, neurons):
  # fix random seed for reproducibility
  seed = 10
  np.random.seed(seed)

  model = Sequential()
  model.add(LSTM(neurons, activation=activation, input_shape=(config.INPUTDAYS, len(config.FEATURES)), return_sequences=False))
  model.add(Dense(units=neurons, activation=activation))
  model.add(Dense(units=1, activation='linear'))
  # Compiling the RNN
  optim = Adam(lr=learn_rate)
  model.compile(optimizer = optim, loss = 'mean_squared_error')
  #model.summary()
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
      epochs=config.EPOCHS,
      validation_data=validation,
      callbacks=[model_checkpoint_callback]
    )
  else:
    train_history = model.fit(X_train, Y_train, epochs=config.EPOCHS, validation_data=validation)
  return pd.DataFrame.from_dict(train_history.history)


def calculate_shap(model: Sequential, X_train: np.ndarray, X_test: np.ndarray, features: List[str]):
  plt.close('all')
  explainer = shap.DeepExplainer(model, X_train)
  shap_values = explainer.shap_values(X_test)
  shap.initjs()
  shap_values_2d = shap_values[0].reshape(-1, len(config.FEATURES))
  X_test_2d = X_test.reshape(-1, len(config.FEATURES))

  shap.summary_plot(shap_values_2d[:, :len(config.FEATURES) - 1], X_test_2d[:, :len(config.FEATURES) - 1],
                    features[:-1])


def predict(model: Sequential, x: np.ndarray, days: int, series_dim: int = -1):
  predictions = np.array([])
  rec_x = x.copy()

  # Make recursive predictions.
  for day in range(days):
    pred = model.predict(rec_x[day:day + config.INPUTDAYS].reshape(1, config.INPUTDAYS, len(config.FEATURES)))
    predictions = np.append(predictions, pred)
    rec_x[day + config.INPUTDAYS + 1, series_dim] = pred
    # rec_x = np.append(rec_x[1:], pred)
  return predictions
