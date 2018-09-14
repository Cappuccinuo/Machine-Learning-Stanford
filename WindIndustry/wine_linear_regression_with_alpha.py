import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utilities import Utilities
np.set_printoptions(suppress=True)

class wine_regression:
  func = Utilities()
  wine_data = pd.read_csv('wine_quality_train.csv')  # 1300 * 12
  test_data = pd.read_csv('wine_quality_test.csv')   # 299 * 12

  features = ['fixed acidity', 'volatile acidity', 'citric acid',
              'residual sugar', 'chlorides', 'free sulfur dioxide',
              'total sulfur dioxide', 'density', 'pH', 'sulphates',
              'alcohol']  # 11 features
  f = np.array(wine_data[features])
  t = np.array(test_data[features])
  Y = np.array(wine_data['quality'])
  TY = np.array(test_data['quality'])

  m = len(f)
  x0 = np.ones(m)
  X = np.insert(f, 0, values=x0, axis=1)
  C = np.zeros(12)
  A = np.ones(1300)
  alpha = 0.0003

  test_len = len(t)
  tx0 = np.ones(test_len)
  TX = np.insert(t, 0, values=tx0, axis=1)

  initial_cost = func.cost_function_with_alpha(X, Y, C, A)

  newC, newA, cost_history = func.gradient_descent_with_alpha(X, Y, C, A, alpha, 1000000)

  Y_pred = TX.dot(newC)
  print(np.round(newC, decimals=4))
  print(func.rmse(TY, Y_pred))
  print(func.r2_score(TY, Y_pred))

  fig = plt.subplots(figsize=(10, 10), dpi=300)
  y_range = np.arange(len(newC[1:]))
  plt.bar(y_range, newC[1:], align='center', alpha=0.5)
  for a, b in zip(y_range, newC[1:]):
    plt.text(a, b, np.round(b, decimals=4), horizontalalignment='center')
  plt.xticks(y_range, features, rotation=30)
  plt.ylabel('Linear Regression Model Parameters')
  plt.show()




