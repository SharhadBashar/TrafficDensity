'''
Author: Sharhad Bashar
Class: Trainer
Description: This class trains on the dataset and creates the models for each stage. It then saves the trained models. For prediction, a data and time is passed.
             The function can load saved models and then predict the out put and return it.
             Input: Name of models to be saved after training. Date and time for prediction
             Output: Trained models. Outputs of predictions
'''

import os
import pickle
import joblib
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from map import map
from util import Util


class Trainer:
  def __init__(self, stage):
    self.root_folder = '..'
    self.model_folder = self.root_folder + '/Model/'
    self.data_folder = self.root_folder + '/Data/'
    self.util = Util()
    # self.model_stage1 = BaggingRegressor(base_estimator = RandomForestRegressor(n_estimators = 1000))
    self.model_stage1 = RandomForestRegressor(n_estimators = 1000)
    self.model_stage2 = ExtraTreesRegressor(n_estimators = 1000)

    if not ((os.path.isfile(self.model_folder + 'stage1.pkl')) and (os.path.isfile(self.model_folder + 'stage2.pkl'))):
      if (stage == 1):
      	self.trained_stage1 = self.learning_stage1()
      	# self.save_model(self.trained_stage1, 'stage1' + datetime.datetime.now() + '.pkl')
      	self.save_model(self.trained_stage1, 'stage1.pkl')
      elif (stage == 2):
      	self.trained_stage2 = self.learning_stage2()
      	# self.save_model(self.trained_stage2, 'stage2' + datetime.datetime.now() + '.pkl')
      	self.save_model(self.trained_stage2, 'stage2.pkl')

  def learning_stage1(self):
    print('Starting stage 1 training')
    X, y = self.util.get_training_data_stage1(self.data_folder)
    model_stage1 = self.model_stage1
    model_stage1.fit(X, y)
    print('Done Stage 1 training')
    return model_stage1

  def learning_stage2(self):
    print('Starting stage 2 training')
    X, y = self.util.get_training_data_stage2(self.data_folder)
    model_stage2 = self.model_stage2
    model_stage2.fit(X, y)
    print('Done Stage 2 training')
    return model_stage2

  def predict(self, date, time, stage1 = 'stage1.pkl', stage2 = 'stage2.pkl'):
    # intersections = Data_Processing().get_intersections()
    util = self.util
    output = {}
    intersections = map.keys()
    year, month, day = date.split('-')
    start_hour, start_minute, end_hour, end_minute = util.get_time_range(time)
    is_weekend = util.is_weekend(date)
    is_holiday = util.is_holiday(date)

    model_stage1 = self.load_model(self.model_folder + stage1)
    model_stage2 = self.load_model(self.model_folder + stage2)


    for intersection in intersections:
      stage1 = model_stage1.predict(pd.DataFrame({'location_id': intersection, 'year': year, 'month': month, 'day': day, 'time_start_hour': start_hour,
                   'time_start_min': start_minute, 'time_end_hour': end_hour, 'time_end_min': end_minute, 'num_lanes': util.get_lanes(intersection),
                   'is_oneway': util.get_oneway(intersection), 'is_weekend': int(is_weekend), 'is_holiday': int(is_holiday)}, index = [0]))

      stage2 = model_stage2.predict(pd.DataFrame({'location_id': intersection, 'year': year, 'month': month, 'day': day, 'time_start_hour': start_hour,
                   'time_start_min': start_minute, 'time_end_hour': end_hour, 'time_end_min': end_minute, 'num_lanes': util.get_lanes(intersection),
                   'is_oneway': util.get_oneway(intersection), 'is_weekend': int(is_weekend), 'is_holiday': int(is_holiday),
                   'nx': stage1[0][0], 'sx': stage1[0][1], 'ex': stage1[0][2], 'wx': stage1[0][1]}, index = [0]))

      output[intersection] = stage2
    return output

  def predict_end_to_end(self, year, month, day, start_hour, start_minute, end_hour, end_minute, is_weekend, is_holiday, stage1 = 'stage1.pkl', stage2 = 'stage2.pkl'):
    util = self.util
    output = {}
    intersections = map.keys()

    model_stage1 = self.load_model(self.model_folder + stage1)
    model_stage2 = self.load_model(self.model_folder + stage2)


    for intersection in intersections:
      stage1 = model_stage1.predict(pd.DataFrame({'location_id': intersection, 'year': year, 'month': month, 'day': day, 'time_start_hour': start_hour,
                   'time_start_min': start_minute, 'time_end_hour': end_hour, 'time_end_min': end_minute, 'num_lanes': util.get_lanes(intersection),
                   'is_oneway': util.get_oneway(intersection), 'is_weekend': int(is_weekend), 'is_holiday': int(is_holiday)}, index = [0]))

      stage2 = model_stage2.predict(pd.DataFrame({'location_id': intersection, 'year': year, 'month': month, 'day': day, 'time_start_hour': start_hour,
                   'time_start_min': start_minute, 'time_end_hour': end_hour, 'time_end_min': end_minute, 'num_lanes': util.get_lanes(intersection),
                   'is_oneway': util.get_oneway(intersection), 'is_weekend': int(is_weekend), 'is_holiday': int(is_holiday),
                   'nx': stage1[0][0], 'sx': stage1[0][1], 'ex': stage1[0][2], 'wx': stage1[0][1]}, index = [0]))

      output[intersection] = stage2
    return output

  def save_model(self, model, model_name = 'model.pkl'):
    joblib.dump(model, open(self.model_folder + model_name, 'wb'))
    print('Model saved at: ' + self.model_folder + model_name)

  def load_model(self, model_name):
    if not (os.path.isfile(self.model_folder + model_name)):
      return ('Model not found')
    return joblib.load(open(self.model_folder + model_name, 'rb'))

if __name__ == 'main':
  Trainer(0)

