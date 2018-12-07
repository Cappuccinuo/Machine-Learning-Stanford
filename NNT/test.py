import pandas as pd

d1 = pd.read_csv('2014-15_valid_features.csv')
d2 = pd.read_csv('2015-16_valid_features.csv')
d3 = pd.read_csv('2016-17_valid_features.csv')
d4 = pd.read_csv('2017-18_valid_features.csv')
combine_csv = pd.concat([d1, d2, d3])
combine_csv.drop(0)
combine_csv.to_csv('all_valid.csv')