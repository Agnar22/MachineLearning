from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from typing import List
import shap
import numpy as np
import pandas as pd
import config
import main


def create_model():
  model = Sequential()
  model.add(LSTM(config.UNITS, input_shape=(config.INPUTDAYS, len(config.FEATURES)), return_sequences=False))
  model.add(Dropout(0.25))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(1, activation='linear'))
  optim = Adam(lr=0.0001)
  # optim = RMSprop()
  model.compile(loss='logcosh', optimizer=optim)
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
      epochs=config.EPOCHS,
      validation_data=validation,
      callbacks=[model_checkpoint_callback]
    )
  else:
    train_history = model.fit(X_train, Y_train, epochs=config.EPOCHS, validation_data=validation)
  return pd.DataFrame.from_dict(train_history.history)


def calculate_shap(model: Sequential, X_train: np.ndarray, X_test: np.ndarray, features: List[str]):
  explainer = shap.DeepExplainer(model, X_train)
  shap_values = explainer.shap_values(X_test)
  shap.initjs()
  shap_values_2d = shap_values[0].reshape(-1, len(config.FEATURES))
  X_test_2d = X_test.reshape(-1, len(config.FEATURES))

  shap.summary_plot(shap_values_2d, X_test_2d, features)



def predict(model: Sequential, x: np.ndarray, days: int):
  predictions = np.array([])

  # Make recursive predictions.
  for day in range(days):
    pred = model.predict(x.reshape(1, config.INPUTDAYS, len(config.FEATURES)))
    predictions = np.append(predictions, pred)
    x = np.append(x[1:], pred)
  return predictions
