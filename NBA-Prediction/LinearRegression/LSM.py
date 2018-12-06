import numpy as np
import pandas as pd


stepSize = 0.00001
path = 'all.csv'
df = pd.read_csv(path)
max = 300000
w = np.zeros((max + 1, 4))
divisor = len(df)
y = np.zeros(divisor)
e = np.zeros(divisor)
n = 0

for k in range(0, max):
  f = df.iloc[n, 1:4].values
  f = np.insert(f, 0, 1)
  y[n] = np.sum(np.dot(w[k], f))
  e[n] = df['rst'].iloc[n] - y[n]
  w[k + 1] = w[k] + np.dot(np.dot(stepSize, e[n]), f)
  n += 1
  if (n == divisor):
    n = 0

wFinal = np.mean(w, axis=0)
print(wFinal)
# [ 0.05080029 -0.02579095  0.006487    0.63562942]