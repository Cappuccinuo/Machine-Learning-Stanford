import numpy as np

class Utilities:
  def test(self):
    print("Hello")

  def cost_function(self, X, Y, C):
    m = len(Y)
    J = np.sum((X.dot(C) - Y) ** 2) / (2 * m)
    return J

  def gradient_descent(self, X, Y, C, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)

    for iteration in range(iterations):
      h = X.dot(C)
      loss = h - Y
      gradient = X.T.dot(loss) / m
      C = C - alpha * gradient
      cost = self.cost_function(X, Y, C)
      cost_history[iteration] = cost

    return C, cost_history

  def rmse(self, Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse

  def r2_score(self, Y, Y_pred):
    mean_y = np.mean(Y);
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2