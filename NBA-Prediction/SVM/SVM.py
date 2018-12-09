import numpy as np
from sklearn.svm import SVC

# myPath = '../Data/PCAFeatures/'
# dataset = np.loadtxt(myPath + 'all_pca_2010-2017.csv', delimiter=',', skiprows=0)
# X = dataset[:, 0:3]
# Y = dataset[:, 3]
# for i in range(0, len(Y)):
#   if (Y[i] == -1):
#     Y[i] = 0
#
# testset = np.loadtxt(myPath + '2016.csv', delimiter=',', skiprows=0)
# X_test = testset[:, 0:3]
# Y_test = testset[:, 3]
# for i in range(0, len(Y_test)):
#   if (Y_test[i] == -1):
#     Y_test[i] = 0
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


print('rbf')
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X, Y)
score = svclassifier.score(X_test, Y_test)
print(score)

p = np.zeros(7)
for j in range(2010, 2017):
  y = j - 2010
  name = myPath + str(j) + '-' + str(j + 1)[2:] + '_valid_features.csv'
  testset = np.loadtxt(name, delimiter=',', skiprows=1)
  X_test = testset[:, 0:8]
  Y_test = testset[:, 8]
  for i in range(0, len(Y_test)):
    if (Y_test[i] == -1):
      Y_test[i] = 0
  p[y] = svclassifier.score(X_test, Y_test)
print(p)
print(np.mean(p))


testset = np.loadtxt(myPath + '2017-18_valid_features.csv', delimiter=',', skiprows=1)
X_test = testset[:, 0:8]
Y_test = testset[:, 8]
for i in range(0, len(Y_test)):
  if (Y_test[i] == -1):
    Y_test[i] = 0

print('linear')
svclassifier = SVC(kernel='linear')
svclassifier.fit(X, Y)
score = svclassifier.score(X_test, Y_test)
print(score)

p = np.zeros(7)
for j in range(2010, 2017):
  y = j - 2010
  name = myPath + str(j) + '-' + str(j + 1)[2:] + '_valid_features.csv'
  testset = np.loadtxt(name, delimiter=',', skiprows=1)
  X_test = testset[:, 0:8]
  Y_test = testset[:, 8]
  for i in range(0, len(Y_test)):
    if (Y_test[i] == -1):
      Y_test[i] = 0
  p[y] = svclassifier.score(X_test, Y_test)
print(p)
print(np.mean(p))



testset = np.loadtxt(myPath + '2017-18_valid_features.csv', delimiter=',', skiprows=1)
X_test = testset[:, 0:8]
Y_test = testset[:, 8]
for i in range(0, len(Y_test)):
  if (Y_test[i] == -1):
    Y_test[i] = 0

print('sigmoid')
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X, Y)
score = svclassifier.score(X_test, Y_test)
print(score)

p = np.zeros(7)
for j in range(2010, 2017):
  y = j - 2010
  name = myPath + str(j) + '-' + str(j + 1)[2:] + '_valid_features.csv'
  testset = np.loadtxt(name, delimiter=',', skiprows=1)
  X_test = testset[:, 0:8]
  Y_test = testset[:, 8]
  for i in range(0, len(Y_test)):
    if (Y_test[i] == -1):
      Y_test[i] = 0
  p[y] = svclassifier.score(X_test, Y_test)
print(p)
print(np.mean(p))
