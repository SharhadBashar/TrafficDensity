'''
Author: Sharhad Bashar
Class: Main_End_To_End
Description: Main class that calls all other classes for the end to end method. Creates data and model folders if required.
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

class Main_End_To_End:
  def __init__(self, train = True):
    if train:
      self._create_folder()
      Data_Processing(end_to_end = True)
      Trainer(1)
      Trainer(2)
      print('Done Training')

  def _create_folder(self):
    if not os.path.exists('../Data'):
      os.makedirs('../Data')
    if not os.path.exists('../Model'):
      os.makedirs('../Model')

Main_End_To_End()