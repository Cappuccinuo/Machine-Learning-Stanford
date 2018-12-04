import os
import time
import math
import numpy as np
import pandas as pd
import xgboost

def make_submission_xgboost():
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
    print('training a xgboosting regressor ...')
    clf = xgboost.XGBRegressor(n_estimators=200, learning_rate=0.08, gamma=0, subsample=0.75,
                               colsample_bytree=1, max_depth=7)
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