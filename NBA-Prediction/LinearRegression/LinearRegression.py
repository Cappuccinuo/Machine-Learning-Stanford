import pandas as pd
import numpy as np

w = np.zeros(4)
# [ 0.1622    0.0071   -0.0709    0.0165]
w[0] = 0.1622
w[1] = 0.0071
w[2] = -0.0709
w[3] = 0.0165

correct = 0
impredictable = 0

start = 2014
end = 2017
base = 2014
p = np.zeros(end - start + 1)

for year in range(start, end + 1):
  path = str(year) + '.csv'
  df = pd.read_csv(path)
  Y = np.zeros(df.shape[0])
  for i in range(0, df.shape[0]):
    f = df.iloc[i, 0:3].values
    f = np.insert(f, 0, 1)
    Y[i] = np.sum(np.dot(np.transpose(w), f))
    if ((Y[i] > 0 and df['rst'].iloc[i] == 1) or (Y[i] < 0 and df['rst'].iloc[i] == -1)):
      correct += 1
    elif (Y[i] == 0):
      impredictable += 1
  pred = (correct + (impredictable / 2)) / df.shape[0]
  correct = 0
  p[year - base] = pred
print(p)
print(np.mean(p))
