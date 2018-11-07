import time
import json

import numpy as np
import pandas as pd

def process_row_training(row):
  x = row['POLYLINE']
  if (len(x) > 4):
    x = np.array(x, ndmin=2)
    data = process_trip(row['TIMESTAMP'])
    data += [len(x)]
  else:
    data = [-1] * 3
  return pd.Series(np.array(data, dtype=float))

def process_row_testing(row):
  data = process_trip(row['TIMESTAMP'])
  return pd.Series(np.array(data, dtype=float))

def process_trip(start_time):
  tt = time.localtime(start_time)
  data = [tt.tm_wday, tt.tm_hour]
  return data


t0 = time.time()
FEATURES = ['wday', 'hour']

print('reading training data ...')
df = pd.read_csv('../data/train.csv', converters={'POLYLINE': lambda x: json.loads(x)})

print('preparing training data ...')
ds = df.apply(process_row_training, axis=1) # apply function to rows
ds.columns = FEATURES + ['len']
df.drop(['ORIGIN_CALL', 'ORIGIN_STAND', 'DAY_TYPE', 'TIMESTAMP', 'POLYLINE', 'TRIP_ID'],
        axis=1, inplace=True)
df['TAXI_ID'] -= np.min(df['TAXI_ID'])
df = df.join(ds)

df = df[df['MISSING_DATA'] == False]
df.drop(['MISSING_DATA'], axis=1, inplace=True)
df.to_csv('../data/train_V1.csv', index=False)

print('reading test data ...')
df = pd.read_csv('../data/test_public.csv')

print('preparing test data ...')
ds = df.apply(process_row_testing, axis=1)
ds.columns = FEATURES
df.drop(['DAY_TYPE','ORIGIN_CALL','ORIGIN_STAND','MISSING_DATA','TIMESTAMP'], axis=1, inplace=True)
df = df.join(ds)
df.to_csv('../data/test_V1.csv', index=False)