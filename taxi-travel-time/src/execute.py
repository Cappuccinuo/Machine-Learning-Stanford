from concatCSV import concat
from submission_combine import combine
from make_submission import make_submission
from make_submission_xgboost import make_submission_xgboost
from data_preprocessing import process
import os

# pre-processing data
process()
# Random Forest
make_submission()
concat()
os.rename('my_submission_V5.csv', 'my_submission_V4.csv')
# xgboost
make_submission_xgboost()
concat()
os.rename('my_submission_V5.csv', 'my_submission_V6.csv')
combine()