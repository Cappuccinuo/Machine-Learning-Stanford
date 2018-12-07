import pandas as pd
myPath = '../Data/ValidFeatures/'
d0 = pd.read_csv(myPath + '2010-11_valid_features.csv')
d1 = pd.read_csv(myPath + '2011-12_valid_features.csv')
d2 = pd.read_csv(myPath + '2012-13_valid_features.csv')
d3 = pd.read_csv(myPath + '2013-14_valid_features.csv')
d4 = pd.read_csv(myPath + '2014-15_valid_features.csv')
d5 = pd.read_csv(myPath + '2015-16_valid_features.csv')
d6 = pd.read_csv(myPath + '2016-17_valid_features.csv')
combine_csv = pd.concat([d0, d1, d2, d3, d4, d5, d6])
combine_csv.to_csv(myPath + 'all_valid_2010-2017.csv', index=False)