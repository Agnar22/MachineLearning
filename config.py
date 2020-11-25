# Data parameters.
DATA_PATH = 'Data/OxCGRT_latest.csv'
COUNTRIES = ['Norway']
INPUTDAYS = 4
VALIDATION_SIZE = 0.2
INFECTED_LOWER = 100
LSTM_MODEL_DIR = 'Models'

# ConfirmedCases has to be the last column as this is used when indexing.
FEATURES = ['ConfirmedCases']

# LSTM parameters.
UNITS = 256
EPOCHS = 1000
BATCH_SIZE = 64
SAVE = False
