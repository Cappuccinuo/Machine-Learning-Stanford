import pandas as pd
import numpy as np

def process_row(row):
  x = row['TRIP_ID']
  data = [int(x[1:])]
  return pd.Series(np.array(data, dtype=float))

def concat():
  da = pd.read_csv('../src/my_submission_new_A_V5.csv')
  db = pd.read_csv('../src/my_submission_new_B_V5.csv')
  dc = pd.read_csv('../src/my_submission_new_C_V5.csv')

  df = da.append(db).append(dc)
  ds = df.apply(process_row, axis=1)
  ds.columns = ['len']
  df.join(ds)
  df['len'] = ds
  data = df.sort_values(by=['len'])
  data.drop(['len'], axis=1, inplace=True)
  data.to_csv('../src/my_submission_V5.csv', index=False)
