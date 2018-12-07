import numpy as np
from sklearn.linear_model import LogisticRegression

myPath = '../Data/ValidFeatures/'
dataset = np.loadtxt(myPath + 'all_valid_2010-2017.csv', delimiter=',', skiprows=1)
X = dataset[:, 0:8]
Y = dataset[:, 8]
for i in range(0, len(Y)):
  if (Y[i] == -1):
    Y[i] = 0

testset = np.loadtxt(myPath + '2017-18_valid_features.csv', delimiter=',', skiprows=1)
X_test = testset[:, 0:8]
Y_test = testset[:, 8]
for i in range(0, len(Y_test)):
  if (Y_test[i] == -1):
    Y_test[i] = 0

lr = LogisticRegression()
lr.fit(X, Y)
score = lr.score(X_test, Y_test)
print(score)   # 66.64%