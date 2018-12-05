import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

start = 2014
end = 2017
base = 2014

for year in range(start, end + 1):
  path = str(year) + '-' + str(year + 1)[2:] + '_' + 'features.csv'
  df = pd.read_csv(path)
  invalidGames = df.index[(df['HTVP'] == -9999) | (df['ATVP'] == -9999) |
                          (df['HLNGP'] == -9999) | (df['ALNGP'] == -9999) |
                          (df['HVP'] == -9999) | (df['VVP'] == -9999) |
                          (df['HPD'] == -9999) | (df['APD'] == -9999)].tolist()
  minGame = max(invalidGames)
  CompleteFeatureGames = df.iloc[minGame + 1 : , 5 : ]

  CompleteFeatureGames = CompleteFeatureGames.reset_index(drop=True)
  df = CompleteFeatureGames.drop(columns=['rst'])
  K = len(df)
  col = df.shape[1]
  sumVec = np.zeros((1, col))
  for i in range(0, len(df.columns)):
    feature = df.columns[i]
    sumVec[0][i] = np.sum(df[feature])
  meanVector = np.transpose(sumVec)

  aux = np.zeros((col, col))
  for i in range(0, K):
    aux = aux + (np.transpose(df.iloc[i, :].values) - meanVector) * np.transpose(np.transpose(df.iloc[i, :].values) - meanVector)
  aux = aux / K

  u, s, vh = np.linalg.svd(aux, full_matrices=False)
  smat = np.zeros((col, col))
  smat[:col, : col] = np.diag(s)

  r = 3
  Vp = u[:, 0 : r]
  Y = np.transpose(np.dot(np.transpose(Vp), np.transpose(df.values)))

  principalDf = pd.DataFrame(data=Y, columns=['pc1', 'pc2', 'pc3'])
  finalDf = pd.concat([principalDf, CompleteFeatureGames['rst']], axis=1)
  fn = str(year) + '-' + str(year + 1)[2:] + '_pca_features.csv'
  finalDf.to_csv(fn)

  # features = ['HTVP', 'ATVP', 'HLNGP', 'ALNGP', 'HVP', 'VVP', 'HPD', 'APD']
  # x = CompleteFeatureGames.loc[:, features].values
  # y = CompleteFeatureGames.loc[:, ['rst']].values
  # x = StandardScaler().fit_transform(x)
  #
  # pca = PCA(n_components=3)
  # principalComponents = pca.fit_transform(x)
  # principalDf = pd.DataFrame(data = principalComponents, columns= ['pc1', 'pc2', 'pc3'])
  # finalDf = pd.concat([principalDf, CompleteFeatureGames['rst']], axis=1)
