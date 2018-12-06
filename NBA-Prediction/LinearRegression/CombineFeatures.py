import pandas as pd

filenames = ['2014-15_pca_features.csv', '2015-16_pca_features.csv',
             '2016-17_pca_features.csv', '2017-18_pca_features.csv']

combine_csv = pd.concat([pd.read_csv(f) for f in filenames])
combine_csv.to_csv('combined_pca_features.csv', index=False)

