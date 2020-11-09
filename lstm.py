# from keras.layers import Dense, LSTM, Dropout
# from keras.models import Sequential
# from keras.optimizers import Adam, RMSprop
# import numpy as np
# import config
# import main


# def create_model():
#   model = Sequential()
#   model.add(LSTM(config.UNITS, input_shape=(config.INPUTDAYS, 1), return_sequences=False))
#   model.add(Dropout(0.25))
#   model.add(Dense(128, activation='relu'))
#   model.add(Dropout(0.25))
#   model.add(Dense(128, activation='relu'))
#   model.add(Dropout(0.25))
#   model.add(Dense(128, activation='relu'))
#   model.add(Dense(1, activation='linear'))
#   optim = Adam(lr=0.0001)
#   #optim = RMSprop()
#   model.compile(loss='mse', optimizer=optim)
#   model.summary()
#   return model


# def train_model(model: Sequential, X_train: np.ndarray, Y_train: np.ndarray, validation=None):
#   model.fit(x=X_train, y=Y_train, epochs=25, batch_size=128, verbose=True, validation_data=validation)


# def predict(model: Sequential, x: np.ndarray, days: int):
#   # Normalize x.
#   x_norm, _, x_max, x_min = main.normalize_data(x)
#   predictions = np.array([])

#   # Make recursive predictions.
#   for day in range(days):
#     pred_norm = model.predict(x_norm.reshape((1, *x_norm.shape, 1)))
#     pred_norm = pred_norm.flatten()
#     pred = main.de_normalize(pred_norm, x_max, x_min)
#     predictions = np.append(predictions, pred)
#     x = main.de_normalize(x_norm, x_max, x_min)
#     x = np.append(x[1:], pred)
#     x_norm, _, x_max, x_min = main.normalize_data(x)
#   return predictions
