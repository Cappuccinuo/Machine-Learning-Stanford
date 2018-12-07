import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

myPath = '../Data/PCAFeatures/'
dataset = np.loadtxt(myPath + 'all_pca_2010-2017.csv', delimiter=',', skiprows=0)
X = dataset[:, 0:3]
Y = dataset[:, 3]

testset = np.loadtxt(myPath + '2017.csv', delimiter=',', skiprows=0)
X_test = testset[:, 0:3]
Y_test = testset[:, 3]

lr = LinearRegression()
lr.fit(X, Y)

prediction = lr.predict(X_test)
print('Coefficients: \n', lr.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, prediction))
print('Variance score: %.2f' % r2_score(Y_test, prediction))