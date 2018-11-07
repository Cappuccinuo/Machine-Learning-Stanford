import numpy as np
import pandas as pd

submissions = ['my_submission_V1.csv','my_submission_V2.csv',
               'my_submission_V3.csv']


df = pd.read_csv('../data/test_new_V5.csv')
for i, fn in enumerate(submissions):
    tmp = pd.read_csv(fn)
    df['F%i' % i] = tmp['TRAVEL_TIME']


print('creating submission 1 ...')
df['TRAVEL_TIME'] = df['F2']
df['TRAVEL_TIME'] = np.mean(df[['F0','F1','F2']], axis=1)

submission = df[['TRIP_ID','TRAVEL_TIME']]
submission.to_csv('final_submission_1.csv', index = False)