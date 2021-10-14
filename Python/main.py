'''
Author: Sharhad Bashar
Class: Main
Description: Main class that calls all other classes. Creates data and model folders if required.
             Input: Date and Time, start and finish point
             Output: Complete Graph
'''

import os
import random

from map import map
from graph import Graph
from trainer import Trainer
from get_time import Get_Time
from shortest_path import Shortest_Path
from data_processing import Data_Processing


class Main:
  def __init__(self, graph_name = 'test.csv', train = True):
    if train: 
      self.create_folder()
      Data_Processing()
      Trainer(1)
      Trainer(2)
      print('Done Training')

  def create_folder(self):
    if not os.path.exists('../Data'):
      os.makedirs('../Data')
    if not os.path.exists('../Model'):
      os.makedirs('../Model')

  @staticmethod
  def predict(date, time, graph_name = 'test.csv'):
    output = Trainer(0).predict(date, time)
    Graph().create_graph(graph_name, output)
    print('Graph for stage 3 created')

  @staticmethod
  def get_path(A, B, graph_name = 'test.csv', draw = False):
    return Shortest_Path(A, B, graph_name, draw).return_path()
  
  @staticmethod
  def get_time(path, graph_name = 'test.csv', has_density = True):
    return Get_Time(path, graph_name, has_density)

# This will train it
Main()

# This will make a prediction for a date and time
Main(train = False).predict('2020-01-24', '23:12')

# This will give the shortest path from A to B with no traffic density values
print('Start:', 13060)
print('End:', 4563)
path_0 = Main(train = False).get_path(13060, 4563, graph_name = 'graph.csv')
print('Without density:')
print(path_0)
print('Time taken:', Main(train = False).get_time(path_0, graph_name = 'graph.csv', has_density = False).zero_density_time(path_0), 'hours')
print('')
print('Without knowing density:')
print('Time taken:', Main(train = False).get_time(path_0).travel_time(path_0), 'hours')
print('')
# This will give the shortest path from A to B with traffic density values
path_1 = Main(train = False).get_path(13060, 4563, draw = True)
print('With density:')
print(path_1)
print('Time taken:', Main(train = False).get_time(path_1).travel_time(path_1), 'hours')

def end_to_end_test():
  intersections = map.keys()

  for i in range(10):
    A, B = random.sample(intersections, 2)
    print('Start:', A)
    print('End:', B)
    print('')
    print('Without density:')
    path_0 = Main(train = False).get_path(A, B, graph_name = 'graph.csv')
    print(path_0)
    print('Time taken:', Main(train = False).get_time(path_0, graph_name = 'graph.csv', has_density = False).zero_density_time(path_0), 'hours')
    print('')
    print('Without knowing density:')
    print('Time taken:', Main(train = False).get_time(path_0).travel_time(path_0), 'hours')
    print('')
    print('With density:')
    path_1 = Main(train = False).get_path(A, B, draw = False)
    print(path_1)
    print('Time taken:', Main(train = False).get_time(path_1).travel_time(path_1), 'hours')

    print('')