from sklearn.metrics import mean_squared_error
import pandas as pd
import os

DATA_DIR = '../src'
submission = pd.read_csv(os.path.join(DATA_DIR, 'my_submission_V1.csv'))
new_df = pd.read_csv(os.path.join(DATA_DIR, 'my_submission_RND.csv'))
y_true = new_df['TRAVEL_TIME']
y_pre = submission['TRAVEL_TIME']
print(mean_squared_error(y_true, y_pre))