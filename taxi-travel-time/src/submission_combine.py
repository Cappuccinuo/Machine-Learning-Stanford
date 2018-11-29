import numpy as np
import pandas as pd

def combine():
  submissions = ['my_submission_V4.csv',
                 'my_submission_V6.csv']


  df = pd.read_csv('../data/test_new_V5.csv')
  for i, fn in enumerate(submissions):
      tmp = pd.read_csv(fn)
      df['F%i' % i] = tmp['TRAVEL_TIME']


  print('creating submission 2 ...')
  df['TRAVEL_TIME'] = df['F1']
  df['TRAVEL_TIME'] = np.mean(df[['F0','F1']], axis=1)

  submission = df[['TRIP_ID','TRAVEL_TIME']]
  submission.to_csv('final_submission.csv', index = False)