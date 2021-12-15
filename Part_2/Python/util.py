'''
Author: Sharhad Bashar
Class: Util
Description: This class has various utility functions required by other classes
             Input: Time. Date
             Output: Formatted time. Formatted Date
'''

import os
import holidays
import datetime as dt
from datetime import date, datetime

class Util:
  def __init__(self):
    self.data_folder = '../Data/'

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
