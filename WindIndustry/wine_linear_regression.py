import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utilities import Utilities
np.set_printoptions(suppress=True)
from sklearn.metrics import r2_score

class wine_regression:
  func = Utilities()
  wine_data = pd.read_csv('wine_quality_train.csv')  # 1300 * 12
  features = ['fixed acidity', 'volatile acidity', 'citric acid',
              'residual sugar', 'chlorides', 'free sulfur dioxide',
              'total sulfur dioxide', 'density', 'pH', 'sulphates',
              'alcohol']  # 11 features
  f = np.array(wine_data[features])
  Y = np.array(wine_data['quality'])

  m = len(f)
  x0 = np.ones(m)

  X = np.insert(f, 0, values=x0, axis=1)
  C = np.zeros(12)
  alpha = 0.0003

  initial_cost = func.cost_function(X, Y, C)

  newC, cost_history = func.gradient_descent(X, Y, C, alpha, 100000)

  Y_pred = X.dot(newC)
  print(np.round(newC, decimals=4)) # [ 0.3455  0.0575 -0.9993 -0.0571 -0.0028 -0.1717  0.0029 -0.0028  0.3385 0.4098  0.6679  0.3202]
  print(func.rmse(Y, Y_pred)) # 0.6548525576726619
  print(func.r2_score(Y, Y_pred)) # 0.35155465270372244




