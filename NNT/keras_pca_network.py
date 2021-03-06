from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy
numpy.random.seed(6)

dataset = numpy.loadtxt('all.csv', delimiter=',')
X = dataset[:, 0:3]
Y = dataset[:, 3]
for i in range(0, len(Y)):
  if (Y[i] == -1):
    Y[i] = 0

testset = numpy.loadtxt('2017.csv', delimiter=',')
X_test = testset[:, 0:3]
Y_test = testset[:, 3]
for i in range(0, len(Y_test)):
  if (Y_test[i] == -1):
    Y_test[i] = 0

# create model
model = Sequential()
model.add(Dense(3, input_dim=3, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
# sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.8, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=500, batch_size=50)

# Evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))