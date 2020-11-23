import config
import pandas as pd
import matplotlib.pyplot as plt
import lstm
from keras.models import Sequential


def visualize_spread_for_countries(data: pd.DataFrame):
  """
  :param data: a pandas dataframe of the data to visualize.
  :return:
  """
  countries_to_visualize = []
  for country in config.COUNTRIES:
    countries_to_visualize.append(
      {
        'x': data[data['CountryName'] == country]['date'],
        'y': data[data['CountryName'] == country]['total_cases_per_million'],
        'name': country
      }
    )
  draw_graph(*countries_to_visualize, x='date', y='total cases per million')


def draw_graph(*args, x: str = 'x', y: str = 'y'):
  """
  :param args: dict('x' : list, 'y' : list, 'name' : str)
  :param y: label for y axis.
  :param x: label for x axis.
  :return:
  """

  plt.close('all')
  for func in args:
    plt.plot(func['x'], func['y'], label=func['name'])
  plt.xlabel(x)
  plt.ylabel(y)
  plt.xticks(fontsize=8)
  plt.legend()
  plt.show(block=False)


def visualize_predictions(cases: pd.DataFrame, model: Sequential, cases_norway: pd.DataFrame):
  loop = True
  while loop:
    try:
      start_day = int(input("Start day:"))
      prediction_length = int(input("prediction length:"))
      output_start = start_day + config.INPUTDAYS
      output_end = output_start + prediction_length
      features = ['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events',
                  'C4_Restrictions on gatherings',
                  'C5_Close public transport', 'C6_Stay at home requirements', 'C7_Restrictions on internal movement',
                  'C8_International travel controls', 'E1_Income support', 'E2_Debt/contract relief',
                  'E3_Fiscal measures',
                  'E4_International support', 'H1_Public information campaigns', 'H2_Testing policy',
                  'H3_Contact tracing',
                  'H4_Emergency investment in healthcare', 'H6_Facial Coverings', 'ConfirmedCases']

      predictions = lstm.predict(model, cases.iloc[start_day:output_start][features].to_numpy(),
                                 prediction_length)
      for day in range(predictions.shape[0]):
        print(cases['date'].iloc[output_start + day], predictions[day], cases['ConfirmedCases'].iloc[output_start + day])
      draw_graph(
        {'x': cases['date'].iloc[output_start:output_end], 'y': predictions.tolist(), 'name': 'prediction'},
        {'x': cases['date'].iloc[:start_day], 'y': cases_norway['ConfirmedCases'].iloc[:start_day], 'name': 'start'},
        {'x': cases['date'].iloc[start_day:output_start], 'y': cases['ConfirmedCases'].iloc[start_day:output_start],
         'name': 'input'},
        {'x': cases['date'].iloc[output_start:output_end], 'y': cases['ConfirmedCases'].iloc[output_start:output_end],
         'name': 'target'},
      )
    except:
      ans = input("quit?")
      if ans == 'y':
        loop = False
