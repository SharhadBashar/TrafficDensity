import pandas as pd
import numpy as np
from datetime import date, datetime
import holidays

lanes = pd.read_csv('tmcs_2020_2029_lanes.csv', index_col = 0)

data = pd.read_csv('tmcs_2020_2029.csv')

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

row_dict = {
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
data_list = []

def getTime(time):
    time = time.split(' ')[1].split('-')[0]
    hour, minute, _ = time.split(':')
    return float(hour), float(minute)

def isWeekend(date):
    return datetime.strptime(date, '%Y-%m-%d').weekday() > 4

def isHoliday(date):
    return datetime.strptime(date, '%Y-%m-%d') in holidays.CA()

for index, row in data.iterrows():
    row_dict['location_id'] = row['location_id']
    row_dict['year'], row_dict['month'], row_dict['day'] = row['count_date'].split('-')
    row_dict['time_start_hour'], row_dict['time_start_min'] = getTime(row['time_start'])
    row_dict['time_end_hour'], row_dict['time_end_min'] = getTime(row['time_end'])
    row_dict['num_lanes'] = row['lanes']
    row_dict['is_oneway'] = row['is_oneway']
    row_dict['is_weekend'] = isWeekend(row['count_date'])
    row_dict['is_holiday'] = isHoliday(row['count_date'])

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
    data_list.append(row_dict)

finalData = pd.DataFrame(data_list)
finalData.to_csv('tmcs_2020_2029_final.csv', index=False)