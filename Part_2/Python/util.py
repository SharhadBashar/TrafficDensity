'''
Author: Sharhad Bashar
Class: Util
Description: This class has various utility functions required by other classes
             Input: Time. Date
             Output: Formatted time. Formatted Date
'''

import os
import holidays
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date, datetime

from map import map

class Util:
  def __init__(self, lanes = 'tmcs_2020_2029_lanes.csv', final_data = 'tmcs_2020_2029_clean.csv', final_data_end_to_end = 'tmcs_2020_2029_end_to_end.csv'):
    self.data_folder = '../Data/'
    self.lanes = pd.read_csv(self.data_folder + lanes, index_col = 0)
    self.final_data = final_data

  def get_time(self, time):
    hour, minute, _ = time.split(' ')[1].split('-')[0].split(':')
    return float(hour), float(minute)

  def increment_time(self, hour, minute, increment_value = 15):
    return datetime.strptime(str(int(hour)) + ':' + str(int(minute)), '%H:%M') + dt.timedelta(minutes = increment_value)

  def get_time_range(self, time):
    # time in format -> hh:mm
    hh = int(time.split(':')[0])
    mm = int(time.split(':')[1])
    if (mm >= 0 and mm <= 15):
      start_minute = 0
      end_minute = 15
      end_hour = hh
    elif (mm >= 15 and mm <= 30):
      start_minute = 15
      end_minute = 30
      end_hour = hh
    elif (mm >= 30 and mm <= 45):
      start_minute = 30
      end_minute = 45
      end_hour = hh
    else:
      start_minute = 45
      end_minute = 0
      end_hour = (hh + 1) % 24
    return(hh, start_minute, end_hour, end_minute)

  def is_weekend(self, date):
    return datetime.strptime(date, '%Y-%m-%d').weekday() > 4

  def is_holiday(self, date):
    return datetime.strptime(date, '%Y-%m-%d') in holidays.CA()

  def get_directory(self, nav_up = 0):
    for i in range(nav_up):
      os.chdir("..")
    return os.path.abspath(os.curdir)

  def delete_graph(self, graph_name):
    if os.path.exists(self.data_folder + graph_name):
      os.remove(self.data_folder + graph_name)

  def get_intersections(self):
    return self.lanes.index.values

  def get_lanes(self, intersection):
    return self.lanes.loc[intersection]['lanes']

  def get_oneway(self, intersection):
    if (np.isnan(self.lanes.loc[intersection]['one_way'])): return 0
    else: return 1

  def get_training_data_stage1(self, path):
    data = shuffle(pd.read_csv(path + self.final_data))
    data["is_oneway"] = data["is_oneway"].astype(int)
    data["is_weekend"] = data["is_weekend"].astype(int)
    data["is_holiday"] = data["is_holiday"].astype(int)
    X = data[['location_id', 'year', 'month', 'day', 'time_start_hour',
       'time_start_min', 'time_end_hour', 'time_end_min', 'num_lanes',
       'is_oneway', 'is_weekend', 'is_holiday']]
    y = data[['nx', 'sx', 'ex', 'wx']]
    return X, y

  def get_training_data_stage2(self, path):
    data = shuffle(pd.read_csv(path + self.final_data))
    data["is_oneway"] = data["is_oneway"].astype(int)
    data["is_weekend"] = data["is_weekend"].astype(int)
    data["is_holiday"] = data["is_holiday"].astype(int)
    X = data[['location_id', 'year', 'month', 'day', 'time_start_hour',
       'time_start_min', 'time_end_hour', 'time_end_min', 'num_lanes',
       'is_oneway', 'is_weekend', 'is_holiday', 'nx', 'sx', 'ex', 'ex']]
    y = data[['nb_r', 'nb_t', 'nb_l', 'sb_r', 'sb_t', 'sb_l', 'eb_r', 'eb_t', 'eb_l', 'wb_r', 'wb_t', 'wb_l']]
    return X, y

  def get_training_data_end_to_end(self, path):
    data = shuffle(pd.read_csv(path + self.final_data_end_to_end))
    data["is_weekend"] = data["is_weekend"].astype(int)
    data["is_holiday"] = data["is_holiday"].astype(int)
    X = data[['year', 'month', 'day', 'time_start_hour',
       'time_start_min', 'time_end_hour', 'time_end_min',
       'is_weekend', 'is_holiday']]
    y = data[['path']]
    return X, y

  def _get_intersections(self, intersection):
    intersections = []
    if (intersection['N'] != []): intersections.append(intersection['N'][0])
    if (intersection['E'] != []): intersections.append(intersection['E'][0])
    if (intersection['S'] != []): intersections.append(intersection['S'][0])
    if (intersection['W'] != []): intersections.append(intersection['W'][0])
    return intersections

  def valid_path(self, path):
    for i in range(len(path) - 1):
      current_intersection = path[i]
      intersections = self._get_intersections(map[current_intersection])
      next_intersection = path[i + 1]
      if (next_intersection not in intersections): return False
    return True

  def calc_accuracy(self, actual, pred):
    if (len(pred) == 0 or pred[0] != actual[0] or len(pred) > len(actual)):
      return 0
    if (pred == actual):
      return 1
    match = 0
    for i in range(len(pred)):
      if (pred[i] == actual[i]):
        match += 1
      else: break
    return match/len(actual)








