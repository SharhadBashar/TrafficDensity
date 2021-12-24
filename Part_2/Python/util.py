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
from statistics import mean
from datetime import date, datetime

from map import map
from get_time import Get_Time

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
    if (len(pred) == 0 or pred[0] != actual[0] or len(pred) > len(actual) or  not self.valid_path(pred)):
      return 0
    if (pred == actual):
      return 1
    match = 0
    for i in range(len(pred)):
      if (pred[i] == actual[i]):
        match += 1
      else: break
    return match/len(actual)

  def report(self, actual, pred):
    '''
    valid path means all intersections are in a correct order
    correct path means a valid path connecting start to end

    pred_len_0 -> len of prediction is 0
    wrong_start_point -> prediction has the wrong start point
    pred_not_valid -> prediction is not a valid path (2 intersections that are not connected come after one another)
    pred_same_actual -> prediction is the same as actual
    pred_correct_different -> prediction is a correct valid path, but shorter than or equal to actual
    pred_correct_longer -> prediction is a correct valid path, but longer than actual
    pred_valid_wrong_end -> prediction is a valid path or length shorter or equal to actual, but has the wrong end point
    pred_valid_longer_wrong_end -> prediction is a valid path, but longer than actual and has the wrong end point
    '''

    pred_len_0 = 0
    wrong_start_point = 0
    pred_not_valid = 0
    pred_same_actual = 0
    pred_correct_different, pred_correct_different_time, pred_correct_different_distance = 0, [], []
    pred_correct_longer, pred_correct_longer_time, pred_correct_longer_distance = 0, [], []
    pred_valid_wrong_end, pred_valid_wrong_end_match = 0, []
    pred_valid_longer_wrong_end = 0

    for i in range(len(pred)):
      if (len(pred)) == 0:
        pred_len_0 += 1
        continue

      if (actual[0] != pred[0]):
        wrong_start_point += 1
        continue

      if (not self.valid_path(pred)):
        pred_not_valid += 1
        continue

      if (actual == pred):
        pred_same_actual += 1
        continue

      if (self.valid_path(pred) and pred[-1] == actual[-1] and len(pred) <= len(actual)):
        pred_correct_different += 1
        pred_correct_different_time.append(Get_Time(pred).travel_time(pred) - Get_Time(actual).travel_time(actual))
        continue

      if (self.valid_path(pred) and pred[-1] == actual[-1] and len(pred) > len(actual)):
        pred_correct_longer += 1
        pred_correct_longer_time.append(Get_Time(pred).travel_time(pred) - Get_Time(actual).travel_time(actual))
        pred_correct_longer_distance.append(len(pred) - len(actual))
        continue

      if (self.valid_path(pred) and len(pred) <= len(actual) and pred[-1] != actual[-1]):
        pred_valid_wrong_end += 1
        match = 0
        for i in range(len(pred)):
          if (pred[i] == actual[i]):
            match += 1
          else: break
        pred_valid_wrong_end_match.append(match / len(actual))
        continue

      if (self.valid_path(pred) and len(pred) > len(actual) and pred[-1] != actual[-1]):
        pred_valid_longer_wrong_end += 1

    print('Total test cases:', len(pred))
    print('Length of prediction is 0 for {} cases'.format(pred_len_0))
    print('Start point is not the same for {} cases'.format(wrong_start_point))
    print('Prediction is not a valid path for {} cases'.format(pred_not_valid))
    print('Prediction is the same as actual for {} cases'.format(pred_same_actual))
    print('Prediction is a correct valid path, but not the same as actual (shorter or equal length) for {} cases. Average travel time is {}s more than actual'.format(pred_correct_different, mean(pred_correct_different_time)))
    print('Prediction is a correct valid path, but longer than actual for {} cases. Average travel time is {}s more than actual. Average travel distance is {}km more than actual'.format(pred_correct_longer, mean(pred_correct_longer_time), mean(pred_correct_longer_distance)))
    print('Prediction is a valid but incorrect (wrong end point) path for {} cases. Prediction matches {} % on average with actual'.format(pred_valid_wrong_end, mean(pred_valid_wrong_end_match) * 100))
    print('Prediction is a valid but incorrect and longer path for {} cases'.format(pred_valid_longer_wrong_end))











