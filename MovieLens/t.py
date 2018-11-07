import numpy as np

Y_test = np.loadtxt('test.txt')	.astype(int)

print(Y_test[:,2])

# for i in range(len(Y_test)):
#   print(Y_test[i][2])