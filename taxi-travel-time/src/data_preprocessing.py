import time
import math
import json
import numpy as np
import pandas as pd

def process_row_training(row):
  x = row['POLYLINE']
  if (len(x) >= 4):
    x = np.array(x, ndmin=2)
    data = process_train_trip(x[0, :], row['TIMESTAMP'])
    data += [x[-1, 0], x[-1, 1], len(x)]
  else:
    data = [-1] * 10
  return pd.Series(np.array(data, dtype=float))

def process_train_trip(x, start_time):
  tt = time.localtime(start_time)
  # data = [tt.tm_wday, tt.tm_hour, x[0], x[1]]
  data = [tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_wday, tt.tm_hour]
  weekend = 0
  if (tt.tm_wday == 5 or tt.tm_wday == 6):
    weekend = 1
  data += [weekend]
  workday = 0
  if (weekend == 0 and tt.tm_hour >= 8 and tt.tm_hour <= 18):
    workday = 1
  data += [workday]
  # timescope = 0
  # if (tt.tm_hour >= 0 and tt.tm_hour <= 4):
  #   timescope = 1
  # if (tt.tm_hour > 4 and tt.tm_hour <= 8):
  #   timescope = 2
  # if (tt.tm_hour > 8 and tt.tm_hour <= 12):
  #   timescope = 3
  # if (tt.tm_hour > 12 and tt.tm_hour <= 16):
  #   timescope = 4
  # if (tt.tm_hour > 16 and tt.tm_hour <= 20):
  #   timescope = 5
  # if (tt.tm_hour > 20 and tt.tm_hour <= 23):
  #   timescope = 6
  # data += [timescope]
  data += [x[0], x[1]]
  return data

def process_row_testing(row):
  data = process_test_trip(row['TIMESTAMP'])
  return pd.Series(np.array(data, dtype=float))

def process_test_trip(start_time):
  tt = time.localtime(start_time)
  data = [tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_wday, tt.tm_hour]
  weekend = 0
  if (tt.tm_wday == 5 or tt.tm_wday == 6):
    weekend = 1
  data += [weekend]
  workday = 0
  if (weekend == 0 and tt.tm_hour >= 8 and tt.tm_hour <= 18):
    workday = 1
  data += [workday]
  # timescope = 0
  # if (tt.tm_hour >= 0 and tt.tm_hour <= 2):
  #   timescope = 1
  # if (tt.tm_hour > 2 and tt.tm_hour <= 4):
  #   timescope = 2
  # if (tt.tm_hour > 4 and tt.tm_hour <= 6):
  #   timescope = 3
  # if (tt.tm_hour > 6 and tt.tm_hour <= 8):
  #   timescope = 4
  # if (tt.tm_hour > 8 and tt.tm_hour <= 10):
  #   timescope = 5
  # if (tt.tm_hour > 10 and tt.tm_hour <= 12):
  #   timescope = 6
  # if (tt.tm_hour > 12 and tt.tm_hour <= 14):
  #   timescope = 7
  # if (tt.tm_hour > 14 and tt.tm_hour <= 16):
  #   timescope = 8
  # if (tt.tm_hour > 16 and tt.tm_hour <= 18):
  #   timescope = 9
  # if (tt.tm_hour > 18 and tt.tm_hour <= 20):
  #   timescope = 10
  # if (tt.tm_hour > 20 and tt.tm_hour <= 22):
  #   timescope = 11
  # if (tt.tm_hour > 22 and tt.tm_hour <= 24):
  #   timescope = 12
  # data += [timescope]
  return data

FEATURES_Train = ['year', 'mon', 'mday', 'wday', 'hour', 'weekend', 'workday', 'xs', 'ys', 'xe', 'ye']
FEATURES = ['year', 'mon', 'mday', 'wday', 'hour', 'weekend', 'workday']

print('reading training data ...')
df = pd.read_csv('../data/train.csv', converters={'POLYLINE': lambda x: json.loads(x)})
print('preparing training data ...')
ds = df.apply(process_row_training, axis=1) # apply function to rows
ds.columns = FEATURES_Train + ['len']
df.drop(['TRIP_ID', 'DAY_TYPE', 'TIMESTAMP', 'POLYLINE'], axis=1, inplace=True)
df['TAXI_ID'] -= np.min(df['TAXI_ID'])
df = df.join(ds)
df = df[df['MISSING_DATA'] == False]
df.drop(['MISSING_DATA'], axis=1, inplace=True)
df.to_csv('../data/train_new_V5.csv', index=False)


print('reading test data ...')
df = pd.read_csv('../data/test_public.csv')

ds = df.apply(process_row_testing, axis=1)
ds.columns = FEATURES
df.drop(['DAY_TYPE', 'MISSING_DATA','TIMESTAMP'], axis=1, inplace=True)
df['TAXI_ID'] -= np.min(df['TAXI_ID'])
df = df.join(ds)
df.to_csv('../data/test_new_V5.csv', index=False)

df = pd.read_csv('../data/train_new_V5.csv')
da = df[df['CALL_TYPE'] == 'A']
da.drop(['ORIGIN_STAND'], axis=1, inplace=True)
db = df[df['CALL_TYPE'] == 'B']
db.drop(['ORIGIN_CALL'], axis=1, inplace=True)
db = db[pd.isnull(df['ORIGIN_STAND']) == False]
dc = df[df['CALL_TYPE'] == 'C']
dc.drop(['ORIGIN_STAND', 'ORIGIN_CALL'], axis=1, inplace=True)
da.to_csv('../data/train_new_A_V5.csv', index=False)
db.to_csv('../data/train_new_B_V5.csv', index=False)
dc.to_csv('../data/train_new_C_V5.csv', index=False)

df = pd.read_csv('../data/test_new_V5.csv')
da = df[df['CALL_TYPE'] == 'A']
da.drop(['ORIGIN_STAND'], axis=1, inplace=True)
db = df[df['CALL_TYPE'] == 'B']
db.drop(['ORIGIN_CALL'], axis=1, inplace=True)
dc = df[df['CALL_TYPE'] == 'C']
dc.drop(['ORIGIN_STAND', 'ORIGIN_CALL'], axis=1, inplace=True)
da.to_csv('../data/test_new_A_V5.csv', index=False)
db.to_csv('../data/test_new_B_V5.csv', index=False)
dc.to_csv('../data/test_new_C_V5.csv', index=False)