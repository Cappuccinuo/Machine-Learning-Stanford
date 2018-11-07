import os
import time
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from util import haversineKaggle

DATA_DIR = '../data'
t0 = time.time()
for filename in ['train_new_V5.csv']:
  print('reading training data from %s ...' % filename)

  df = pd.read_csv(os.path.join(DATA_DIR, filename))
  df = df[df['len'] != -1]
  df = df[df['hour'] != -1]

  d1 = haversineKaggle(df[['xs', 'ys']].values, df[['xe', 'ye']].values)

  y = np.log(df['len']*15 + 1)
  df.drop(['year', 'CALL_TYPE', 'len', 'xs', 'ys', 'xe', 'ye'], axis=1, inplace=True)

  # if (filename == 'train_new_A_V1.csv'):
  #   df.drop(['ORIGIN_STAND'], axis=1, inplace=True)
  # if (filename == 'train_new_B_V1.csv'):
  #   df.drop(['ORIGIN_CALL'], axis=1, inplace=True)
  # if (filename == 'train_new_C_V1.csv'):
  #   df.drop(['ORIGIN_STAND', 'ORIGIN_CALL'], axis=1, inplace=True)
  X = np.array(df, dtype=np.float)
  th1 = np.percentile(d1, [99.9])
  X = X[(d1<th1), :]
  y = y[(d1<th1)]

  print('training a random forest regressor ...')
  clf = RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=21)
  clf.fit(X, y)

  print('predicting test data ...')
  df = pd.read_csv(os.path.join(DATA_DIR, filename.replace('train', 'test')))
  ids = df['TRIP_ID']

  # if (filename == 'train_new_A_V1.csv'):
  #   df.drop(['ORIGIN_STAND'], axis=1, inplace=True)
  # if (filename == 'train_new_B_V1.csv'):
  #   df.drop(['ORIGIN_CALL'], axis=1, inplace=True)
  # if (filename == 'train_new_C_V1.csv'):
  #   df.drop(['ORIGIN_STAND', 'ORIGIN_CALL'], axis=1, inplace=True)

  df = df.drop(['CALL_TYPE', 'TRIP_ID', 'year'], axis=1)
  X_tst = np.array(df, dtype=np.float)
  y_pred = clf.predict(X_tst)

  # create submission file
  submission = pd.DataFrame(ids, columns=['TRIP_ID'])
  filename = filename.replace('train', 'my_submission')
  submission['TRAVEL_TIME'] = np.exp(y_pred)
  submission.to_csv(filename, index=False)

print('Done in %.1f sec.' % (time.time() - t0))