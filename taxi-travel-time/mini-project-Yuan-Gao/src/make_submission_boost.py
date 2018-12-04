import os
import time
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


DATA_DIR = '../data'
t0 = time.time()
for filename in ['train_new_A_V5.csv', 'train_new_B_V5.csv', 'train_new_C_V5.csv']:
  print('reading training data from %s ...' % filename)

  df = pd.read_csv(os.path.join(DATA_DIR, filename))
  df = df[df['len'] != -1]
  df = df[df['hour'] != -1]
  y = np.log(df['len']*15 + 1)
  df.drop(['CALL_TYPE', 'len', 'xs', 'ys', 'xe', 'ye'], axis=1, inplace=True)

  X = np.array(df, dtype=np.float)
  print('training a gradient boosting regressor ...')
  clf = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                  random_state=23)
  clf.fit(X, y)

  print('predicting test data ...')
  df = pd.read_csv(os.path.join(DATA_DIR, filename.replace('train', 'test')))
  ids = df['TRIP_ID']

  df = df.drop(['CALL_TYPE', 'TRIP_ID'], axis=1)
  X_tst = np.array(df, dtype=np.float)
  y_pred = clf.predict(X_tst)

  # create submission file
  submission = pd.DataFrame(ids, columns=['TRIP_ID'])
  filename = filename.replace('train', 'my_submission')
  submission['TRAVEL_TIME'] = np.exp(y_pred)
  submission.to_csv(filename, index=False)

print('Done in %.1f sec.' % (time.time() - t0))