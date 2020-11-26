from main import load_and_clean_data
import matplotlib.pyplot as plt
import config

def normalize(data):
  return data/max(data)

def draw_graph_stacked(*args):
  """
  :param args: dict('x' : list, 'y' : list, 'name' : str)
  :param y: label for y axis.
  :param x: label for x axis.
  :return:
  """

  plt.close('all')
  fig, axs = plt.subplots(len(args), sharex=True)
  plt.subplots_adjust(left=None, bottom=None, right=None, top=0.2, wspace=None, hspace=1.0)
  counter = 0
  for func in args:
    axs[counter].plot(func['x'], func['y'])
    axs[counter].set_title(func['name'], fontsize=10)
    counter += 1

def display_policies(health:bool=False):
  cutoff = -1
  data = load_and_clean_data()
  features = list(filter(lambda x: (health and 'H' == x[0]) or (not health and 'C' == x[0]), config.FEATURES))
  test_x = data[data['CountryName'] == 'Norway']['Date'].values[:cutoff]
  test_dict = []
  for feature in features[:-1]:
    test_feature = data[data['CountryName'] == 'Norway'][feature].values[:cutoff]
    test_dict.append({'x':test_x,'y':test_feature,'name':feature})
  draw_graph_stacked(*test_dict)
  plt.show()

if __name__ == '__main__':
  display_policies(True)
  display_policies(False)
