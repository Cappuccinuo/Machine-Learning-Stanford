'''
3rd step
Get the feature with all information in it
'''

import pandas as pd

start = 2010
end = 2017
base = 2010

for year in range(start, end + 1):
  path = '../Data/SeperateFeatures/' + str(year) + '-' + str(year + 1)[2:] + '_' + 'features.csv'
  df = pd.read_csv(path)
  invalidGames = df.index[(df['HTVP'] == -9999) | (df['ATVP'] == -9999) |
                          (df['HLNGP'] == -9999) | (df['ALNGP'] == -9999) |
                          (df['HVP'] == -9999) | (df['VVP'] == -9999) |
                          (df['HPD'] == -9999) | (df['APD'] == -9999)].tolist()
  minGame = max(invalidGames)
  CompleteFeatureGames = df.iloc[minGame + 1 : , 4 : ]
  pn = '../Data/ValidFeatures/' + str(year) + '-' + str(year + 1)[2:] + '_' + 'valid_features.csv'
  CompleteFeatureGames.to_csv(pn, index=False)
