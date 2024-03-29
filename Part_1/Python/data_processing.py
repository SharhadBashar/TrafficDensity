'''
Author: Sharhad Bashar
Class: Data_Processing
Description: This class cleans and sets up the data for learning. There are also various functions for viewing different information about the dataset.
             This class also takes the dataset and returns X and y for both stages of training.
             Input: Names of the csv file
             Output: Cleaned data set
'''

import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from util import Util
from graph import Graph


class Data_Processing:
  def __init__(self, data = 'tmcs_2020_2029.csv', lanes = 'tmcs_2020_2029_lanes.csv', final_data = 'tmcs_2020_2029_clean.csv', final_data_end_to_end = 'tmcs_2020_2029_end_to_end.csv', end_to_end = False):
    self.row_dict = {
      'location_id': 0,
      'year': 0,
      'month': 0,
      'day': 0,
      'time_start_hour': 0,
      'time_start_min': 0,
      'time_end_hour': 0,
      'time_end_min': 0,
      'num_lanes': 0,
      'is_oneway': 0,
      'is_weekend': 0,
      'is_holiday': 0,
      #predict
      'nx': 0,
      'sx': 0,
      'ex': 0,
      'wx': 0,
      #predict
      'nb_r': 0,
      'nb_t': 0,
      'nb_l': 0,
      'sb_r': 0,
      'sb_t': 0,
      'sb_l': 0,
      'eb_r': 0,
      'eb_t': 0,
      'eb_l': 0,
      'wb_r': 0,
      'wb_t': 0,
      'wb_l': 0
    }

    self.row_dict_end_to_end = {
        'year': 0,
        'month': 0,
        'day': 0,
        'time_start_hour': 0,
        'time_start_min': 0,
        'time_end_hour': 0,
        'time_end_min': 0,
        'is_weekend': 0,
        'is_holiday': 0,
        'start': '',
        'end': ''
    }

    self.data_folder = '../Data/'

    self.longest_path = 0
    self.data_list = []
    self.data_list_end_to_end = []
    self.starts_ends_list = []
    self.paths = []

    self.final_data = final_data
    self.final_data_path = self.data_folder + final_data
    self.final_data_end_to_end = final_data_end_to_end
    self.final_data_path_end_to_end = self.data_folder + final_data_end_to_end

    self.data = pd.read_csv(self.data_folder + data)
    self.lanes = pd.read_csv(self.data_folder + lanes, index_col = 0)

    self.util = Util()

    if not (os.path.isfile(self.final_data_path)):
      self.combine_data()
      self.clean_data()
      self.save_data()

      print('Data has been modified, cleaned and saved for learning. Augmented learning data can be found at: ' + self.final_data_path)

    if (end_to_end):
      self.clean_data_end_to_end()
      self.save_data(end_to_end = True)
      self.populate_end_to_end()


  def view_data(self, n = 5, view_type = ''):
    if (view_type == 'columns_original'):
      print(self.data.columns)
    elif (view_type == 'data_original'):
      print(self.data.head(n))
    elif (view_type == 'lanes'):
      print(self.lanes.head(n))
    elif (view_type == 'lanes_columns'):
      print(self.lanes.column)
    elif (view_type == 'lanes_cleaned'):
      final_data = pd.read_csv(self.final_data_path)
      print(self.final_data.columns)
    elif (view_type == 'data_cleaned'):
      pfinal_data = pd.read_csv(self.final_data_path)
      print(self.final_data.head(n))
    else:
      print(self.data.head())

  def get_intersections(self):
    return self.lanes.index.values

  def get_lanes(self, intersection):
    return self.lanes.loc[intersection]['lanes']

  def get_oneway(self, intersection):
    if (np.isnan(self.lanes.loc[intersection]['one_way'])): return 0
    else: return 1


  def combine_data(self):
    data = self.data
    lanes = self.lanes

    lanes_col = []
    oneway_col = []

    for id in data.location_id:
      lanes_col.append(lanes.loc[id].lanes)
      if (np.isnan(lanes.loc[id].one_way)):
        oneway_col.append(False)
      else:
        oneway_col.append(True)

    data.insert(3, 'lanes', lanes_col)
    data.insert(4, 'is_oneway', oneway_col)

    self.data = data

  def clean_data(self):
    data = self.data
    data_list = self.data_list
    row_dict = self.row_dict

    util = self.util

    for index, row in data.iterrows():
      row_dict['location_id'] = row['location_id']
      row_dict['year'], row_dict['month'], row_dict['day'] = row['count_date'].split('-')
      row_dict['time_start_hour'], row_dict['time_start_min'] = util.get_time(row['time_start'])
      row_dict['time_end_hour'], row_dict['time_end_min'] = util.get_time(row['time_end'])
      row_dict['num_lanes'] = row['lanes']
      row_dict['is_oneway'] = row['is_oneway']
      row_dict['is_weekend'] = util.is_weekend(row['count_date'])
      row_dict['is_holiday'] = util.is_holiday(row['count_date'])

      row_dict['nx'] = float(row['nx_peds']) + float(row['nx_bike']) + float(row['nx_other'])
      row_dict['sx'] = float(row['sx_peds']) + float(row['sx_bike']) + float(row['sx_other'])
      row_dict['ex'] = float(row['ex_peds']) + float(row['ex_bike']) + float(row['ex_other'])
      row_dict['wx'] = float(row['wx_peds']) + float(row['wx_bike']) + float(row['wx_other'])

      row_dict['nb_r'] = float(row['nb_cars_r']) + float(row['nb_truck_r']) + float(row['nb_bus_r'])
      row_dict['nb_t'] = float(row['nb_cars_t']) + float(row['nb_truck_t']) + float(row['nb_bus_t'])
      row_dict['nb_l'] = float(row['nb_cars_l']) + float(row['nb_truck_l']) + float(row['nb_bus_l'])

      row_dict['sb_r'] = float(row['sb_cars_r']) + float(row['sb_truck_r']) + float(row['sb_bus_r'])
      row_dict['sb_t'] = float(row['sb_cars_t']) + float(row['sb_truck_t']) + float(row['sb_bus_t'])
      row_dict['sb_l'] = float(row['sb_cars_l']) + float(row['sb_truck_l']) + float(row['sb_bus_l'])

      row_dict['eb_r'] = float(row['eb_cars_r']) + float(row['eb_truck_r']) + float(row['eb_bus_r'])
      row_dict['eb_t'] = float(row['eb_cars_t']) + float(row['eb_truck_t']) + float(row['eb_bus_t'])
      row_dict['eb_l'] = float(row['eb_cars_l']) + float(row['eb_truck_l']) + float(row['eb_bus_l'])

      row_dict['wb_r'] = float(row['wb_cars_r']) + float(row['wb_truck_r']) + float(row['wb_bus_r'])
      row_dict['wb_t'] = float(row['wb_cars_t']) + float(row['wb_truck_t']) + float(row['wb_bus_t'])
      row_dict['wb_l'] = float(row['wb_cars_l']) + float(row['wb_truck_l']) + float(row['wb_bus_l'])

      data_list.append(row_dict.copy())

    self.data_list = data_list

  def save_data(self, end_to_end = False):
    if(end_to_end):
      finalData = pd.DataFrame(self.data_list_end_to_end)
      finalData.to_csv(self.final_data_path_end_to_end, index = False)
    else:
      finalData = pd.DataFrame(self.data_list)
      finalData.to_csv(self.final_data_path, index = False)

  def clean_dataset(self, df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace = True)
    indices_to_keep = df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

  def get_training_data_stage1(self, path):
    data = shuffle(pd.read_csv(path + self.final_data))
    data["is_oneway"] = data["is_oneway"].astype(int)
    data["is_weekend"] = data["is_weekend"].astype(int)
    data["is_holiday"] = data["is_holiday"].astype(int)
    # data = self.clean_dataset(data)
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
    # data = self.clean_dataset(data)
    X = data[['location_id', 'year', 'month', 'day', 'time_start_hour',
       'time_start_min', 'time_end_hour', 'time_end_min', 'num_lanes',
       'is_oneway', 'is_weekend', 'is_holiday', 'nx', 'sx', 'ex', 'ex']]
    y = data[['nb_r', 'nb_t', 'nb_l', 'sb_r', 'sb_t', 'sb_l', 'eb_r', 'eb_t', 'eb_l', 'wb_r', 'wb_t', 'wb_l']]
    return X, y

if __name__ == '__main__':
  Data_Processing()









