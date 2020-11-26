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
    axs[counter].set_title(func['name'], fontsize=10)
    counter += 1
  fig.suptitle('Health system policies')


if __name__ == '__main__':
  cutoff = -1
  data = load_and_clean_data()
  features = ['H1_Public information campaigns', 'H2_Testing policy',
            'H3_Contact tracing', 'H4_Emergency investment in healthcare', 'H5_Investment in vaccines',
            'H6_Facial Coverings', 'ConfirmedCases']
  test_x = data[data['CountryName'] == 'Norway']['Date'].values[:cutoff]
  test_dict = []
  for feature in features[:-1]:
    test_feature = data[data['CountryName'] == 'Norway'][feature].values[:cutoff]
    test_dict.append({'x':test_x,'y':test_feature,'name':feature})
  draw_graph_stacked(*test_dict)
  plt.show()
  