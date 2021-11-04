'''
Author: Sharhad Bashar
Class: Graph
Description: This class creates the graph from the map and output of training. It also includes functions for creating a new graph and saving the graph.
             Input: Name of the graph to be created
             Output: Graph
'''

import os
import math
import numpy as np
import pandas as pd

from map import map
from util import Util

class Graph:
  def __init__(self, graph_name = 'graph.csv'):
    self.graph_name = graph_name

    self.data_folder = '../Data/'
    self.graph_path = self.data_folder + self.graph_name

    if not (os.path.isfile(self.graph_path)):
      self.graph = self.create_new_graph()
      self.save_graph(self.graph, self.graph_name)
      print('Initial graph created and saved at: ' + self.graph_path)

  def create_new_graph(self):
    row = np.full(len(map), fill_value = math.inf)
    row = list(row)
    graph_dict = {}
    for intersection in map:
      graph_dict[intersection] = row
    graph = pd.DataFrame(graph_dict, index = map.keys(), columns = map.keys())

    for intersection in map:
      connection = []
      if (len(map[intersection]['N']) > 0):
          connection.append(map[intersection]['N'][0])
      if (len(map[intersection]['E']) > 0):
          connection.append(map[intersection]['E'][0])
      if (len(map[intersection]['S']) > 0):
          connection.append(map[intersection]['S'][0])
      if (len(map[intersection]['W']) > 0):
          connection.append(map[intersection]['W'][0])
      for direction in connection:
          graph[intersection][direction] = 0

    return graph

  def save_graph(self, graph, graph_name):
    graph.to_csv(self.data_folder + graph_name)


# north = nb_t + eb_r + wb_l = (pred[1] + pred[6] + pred[11]) / map[intersection]['N'][1]
# east  = eb_t + nb_r + sb_l = (pred[7] + pred[0] + pred[5])  / map[intersection]['E'][1]
# south = sb_t + eb_l + wb_r = (pred[4] + pred[8] + pred[9])  / map[intersection]['S'][1]
# west  = wb_t + nb_l + sb_r = (pred[10] + pred[2] + pred[3]) / map[intersection]['W'][1]
# pred =
# 'nb_r': 0,
# 'nb_t': 1,
# 'nb_l': 2,
# 'sb_r': 3,
# 'sb_t': 4,
# 'sb_l': 5,
# 'eb_r': 6,
# 'eb_t': 7,
# 'eb_l': 8,
# 'wb_r': 9,
# 'wb_t': 10,
# 'wb_l': 11


  def create_graph(self, graph_name, prediction):
    graph = pd.read_csv(self.graph_path, index_col = 0)
    for intersection in map:
      pred = prediction[intersection][0]
      if (len(map[intersection]['N']) > 0):
          graph.loc[intersection][str(map[intersection]['N'][0])] = (pred[1] + pred[6] + pred[11]) / map[intersection]['N'][1]

      if (len(map[intersection]['E']) > 0):
          graph.loc[intersection][str(map[intersection]['E'][0])] = (pred[7] + pred[0] + pred[5])  / map[intersection]['E'][1]

      if (len(map[intersection]['S']) > 0):
          graph.loc[intersection][str(map[intersection]['S'][0])] = (pred[4] + pred[8] + pred[9])  / map[intersection]['S'][1]

      if (len(map[intersection]['W']) > 0):
          graph.loc[intersection][str(map[intersection]['W'][0])] = (pred[10] + pred[2] + pred[3]) / map[intersection]['W'][1]

    self.save_graph(graph, graph_name)
    # print('Graph created and saved at: ' + self.data_folder + graph_name)


if __name__ == 'main':
  Graph()


