import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Input : lr Linear Regression Model
#         X_test (n * 30) : 0 1 vectors represent which two teams are playing
#         [1 0 0 0 0 ... 1 0 0 0]
#         Y_test (n * 30) : score of corresponding team in X_test
#         [127 98], [88 92]
def testResult(lr, X_test, Y_test):
  right = 0
  trueDiff = []
  predDiff = []
  for game in range(len(X_test)):
    print("Truth is: %f, %f", Y_test[game][0], Y_test[game][1])
    trueDiff.append(Y_test[game][0] - Y_test[game][1])
    pred = lr.predict([X_test[game]])
    predDiff.append(pred[0][0], pred[0][1])
    print("Prediction is: %f, %f", pred[0][0], pred[0][1])
    if (pred[0][0] - pred[0][1] == 0):
      continue
    test_flag = (Y_test[game][0] - Y_test[game][1]) < 0
    pred_flag = (pred[0][0] - pred[0][1]) < 0
    if (test_flag == pred_flag) :
      right = right + 1
  X = np.arange(len(trueDiff))
  plt.scatter(X, trueDiff, color='black')
  plt.plot(X, predDiff, color='blue', linewidth=3)
  plt.show()
  print("Accuracy is: %f" % float(right / len(X_test)))
  print("Mean Squared error is %.2f" % mean_squared_error(Y_test, pred))
