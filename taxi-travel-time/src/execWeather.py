import pandas as pd
import os
DATA_DIR = '../data'
wdf = pd.read_csv('../data/weather.csv')
wdf.drop(['STATION', 'WT01', 'WT02', 'WT03',
          'WT04', 'WT06', 'WT08', 'WT13', 'WT14', 'WT16', 'WT18', 'WT19', 'WT22'], axis=1, inplace=True)

wdf['tavg'] = (wdf['TMAX'] + wdf['TMIN']) / 2
wdf['snowfall'] = wdf['SNOW']
wdf['precipitation'] = wdf['PRCP']
wdf['snowdepth'] = wdf['SNWD']
wdf.drop(['TMAX', 'TAVG', 'TMIN', 'SNOW', 'PRCP', 'SNWD'], axis=1, inplace=True)

wdf['DATE'] = pd.to_datetime(wdf.DATE, format='%Y-%m-%d')
wdf['year'] = wdf.DATE.dt.year
wdf['mon'] = wdf.DATE.dt.month
wdf['mday'] = wdf.DATE.dt.day
wdf.drop(['DATE'], axis=1, inplace=True)
wdf.to_csv('../data/weather_parse.csv', index=False)


wdf = pd.read_csv('../data/weather_parse.csv')
for filename in ['train_new_A_V5.csv', 'train_new_B_V5.csv', 'train_new_C_V5.csv']:
  df = pd.read_csv(os.path.join(DATA_DIR, filename))
  df = pd.merge(df, wdf, on=['year', 'mon', 'mday'])
  df.to_csv(os.path.join(DATA_DIR, filename), index=False)

  df = pd.read_csv(os.path.join(DATA_DIR, filename.replace('train', 'test')))
  filename = filename.replace('train', 'test')
  df = pd.merge(df, wdf, on=['year', 'mon', 'mday'])
  df.to_csv(os.path.join(DATA_DIR, filename), index=False)