from main import load_and_clean_data
import matplotlib.pyplot as plt

def normalize(data):
  return data/max(data)

def draw_graph_stacked(*args, x: str = 'x', y: str = 'y'):
  """
  :param args: dict('x' : list, 'y' : list, 'name' : str)
  :param y: label for y axis.
  :param x: label for x axis.
  :return:
  """

  plt.close('all')
  fig, axs = plt.subplots(len(args), sharex=True)
  counter = 0
  for func in args:
    axs[counter].plot(func['x'], func['y'])
    counter += 1
  fig.suptitle('Vertically stacked subplots')


if __name__ == '__main__':
  cutoff = -1
  data = load_and_clean_data()
  features = ['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events',
                  'C4_Restrictions on gatherings',
                  'C5_Close public transport', 'C6_Stay at home requirements', 'C7_Restrictions on internal movement',
                  'C8_International travel controls', 'ConfirmedCases']
  test_x = data[data['CountryName'] == 'Norway']['Date'].values[:cutoff]
  test_dict = []
  for feature in features:
    test_feature = data[data['CountryName'] == 'Norway'][feature].values[:cutoff]
    test_dict.append({'x':test_x,'y':test_feature,'name':feature})
  draw_graph_stacked(*test_dict)
  plt.show()
  