# Data parameters.
DATA_PATH = 'Data/OxCGRT_latest.csv'
COUNTRIES = ['Norway']
INPUTDAYS = 21
VALIDATION_SIZE = 0.2
INFECTED_LOWER = 100
LSTM_MODEL_DIR = 'Models'
FEATURES = ['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events', 'C4_Restrictions on gatherings',
            'C5_Close public transport', 'C6_Stay at home requirements', 'C7_Restrictions on internal movement',
            'C8_International travel controls', 'H1_Public information campaigns', 'H2_Testing policy',
            'H3_Contact tracing', 'H4_Emergency investment in healthcare', 'H5_Investment in vaccines',
            'H6_Facial Coverings', 'ConfirmedCases']

# LSTM parameters.
UNITS = 20
EPOCHS = 3
SAVE = True
