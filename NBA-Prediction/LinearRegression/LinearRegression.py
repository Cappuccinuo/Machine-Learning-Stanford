import pandas as pd
import numpy as np
myPath = '../Data/PCAFeatures/'
w = np.zeros(4)

w[0] = 0.0803098784764532
w[1] = -0.0765777889768009
w[2] = 0.0586636125930068
w[3] = -0.0291996591432804

correct = 0
impredictable = 0

start = 2017
end = 2017
base = 2017
p = np.zeros(end - start + 1)

for year in range(start, end + 1):
  path = myPath + str(year) + '.csv'
  # df = pd.read_csv(path)
  df = np.loadtxt(path, delimiter=',')
  Y = np.zeros(df.shape[0])
  for i in range(0, df.shape[0]):
    f = df[i, 0:3]
    f = np.insert(f, 0, 1)
    Y[i] = np.sum(np.dot(np.transpose(w), f))
    if ((Y[i] > 0 and df[i][3] == 1) or (Y[i] < 0 and df[i][3] == -1)):
      correct += 1
    elif (Y[i] == 0):
      impredictable += 1
  pred = (correct + (impredictable / 2)) / df.shape[0]
  correct = 0
  p[year - base] = pred
print(np.mean(p))
