import numpy as np
import matplotlib.pyplot as plt
import itertools

from perceptron_helper import (
  predict,
  plot_data,
  boundary,
  plot_perceptron,
)


def update_perceptron(X, Y, w, b):
  """
  This method updates a perceptron model. Takes in the previous weights
  and returns weights after an update, which could be nothing.

  Inputs:
      X: A (N, D) shaped numpy array containing a single point.
      Y: A (N, ) shaped numpy array containing the labels for the points.
      w: A (D, ) shaped numpy array containing the weight vector.
      b: A float containing the bias term.

  Output:
      next_w: A (D, ) shaped numpy array containing the next weight vector
              after updating on a single misclassified point, if one exists.
      next_b: The next float bias term after updating on a single
              misclassified point, if one exists.
  """
  next_w, next_b = np.copy(w), np.copy(b)

  # ==============================================
  # TODO: Implement update rule for perceptron.
  for index in range(len(X)):
    label = predict(X[index], w, b)
    if (label != Y[index]):
      next_w = w + Y[index] * X[index]
      next_b = b + Y[index]
  # ===============================================

  return next_w, next_b

def run_perceptron(X, Y, w, b, max_iter):
  """
  This method runs the perceptron learning algorithm. Takes in initial weights
  and runs max_iter update iterations. Returns final weights and bias.

  Inputs:
      X: A (N, D) shaped numpy array containing a single point.
      Y: A (D, ) shaped numpy array containing the labels for the points.
      w: A (D, ) shaped numpy array containing the initial weight vector.
      b: A float containing the initial bias term.
      max_iter: An int for the maximum number of updates evaluated.

  Output:
      w: A (D, ) shaped numpy array containing the final weight vector.
      b: The final float bias term.
  """

  # ============================================
  # TODO: Implement perceptron update loop.
  previous_w = w
  previous_b = b
  for index in range(max_iter):
    new_w, new_b = update_perceptron(X, Y, w, b)
    if np.alltrue(new_w == previous_w) and new_b == previous_b:
      return new_w, new_b
    w = new_w
    b = new_b
    previous_w = new_w
    previous_b = new_b
  # =============================================

  return w, b

def run_perceptron(X, Y, w, b, axs, max_iter):
  """
  This method runs the perceptron learning algorithm. Takes in initial weights
  and runs max_iter update iterations. Returns final weights and bias.

  Inputs:
      X: A (N, D) shaped numpy array containing a single point.
      Y: A (N, ) shaped numpy array containing the labels for the points.
      w: A (D, ) shaped numpy array containing the initial weight vector.
      b: A float containing the initial bias term.
      axs: A list of Axes that contain suplots for each timestep.
      max_iter: An int for the maximum number of updates evaluated.

  Output:
      The final weight and bias vectors.
  """

  # ============================================
  # TODO: Implement perceptron update loop.
  previous_w = w
  previous_b = b
  print(w)
  print(b)
  print(X[0])
  print(Y[0])
  print()
  count = 0

  for index in range(max_iter):
    num = index % len(X)
    new_w, new_b = update_perceptron(X, Y, w, b)
    if np.alltrue(new_w == previous_w) and new_b == previous_b:
      return new_w, new_b
    w = new_w
    b = new_b
    print(w)
    print(b)
    print(X[num])
    print(Y[num])
    print()
    previous_w = new_w
    previous_b = new_b
    ax = axs[count]
    count = count + 1
    plot_data(X, Y, ax)
    plot_perceptron(w, b, ax)
  # =============================================

  return w, b

X = np.array([[ -3, -1], [0, 3], [1, -2]])
Y = np.array([ -1, 1, 1])
fig = plt.figure(figsize=(5,4))

weights = np.array([0.0, 1.0])
bias = 0.0

f, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9,8))
axs = list(itertools.chain.from_iterable(ax_arr))
for ax in axs:
    ax.set_xlim(-4.1, 3.1); ax.set_ylim(-3.1, 4.1)

run_perceptron(X, Y, weights, bias, axs, 4)

f.tight_layout()
plt.show()

# X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
# Y = np.array([1, 1, -1, -1])
# fig = plt.figure(figsize=(5,4))
# ax = fig.gca(); ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
# plot_data(X, Y, ax)
#
# weights = np.array([0.0, 1.0])
# bias = 0.0
#
# f, ax_arr = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(9, 8))
# axs = list(itertools.chain.from_iterable(ax_arr))
# for ax in axs:
#   ax.set_xlim(-0.1, 1.1);
#   ax.set_ylim(-0.1, 1.1)
#
# run_perceptron(X, Y, weights, bias, axs, 16)
#
# f.tight_layout()
# plt.show()