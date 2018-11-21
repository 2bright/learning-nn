import numpy as np
import os, sys
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir + '/../')
import library as nn

mnist = nn.datasets.mnist

(x_train, y_train, x_test, y_test) = mnist.load(file_dir + '/../data/mnist')
y_train = mnist.b2d(y_train).astype(float)
y_test = mnist.b2d(y_test).astype(float)

print('train -------------------------------------')
model = nn.models.Sequential(lr = 0.5, loss = 'se', batch_size = 50, layers = [
    nn.layers.Dense(units = 512, input_dim = x_train.shape[0], activation = 'relu'),
    nn.layers.Dense(units = 10, input_dim = 512, activation = 'relu'),
    nn.layers.Dense(units = 1, input_dim = 10, activation = 'linear')
])

model.fit(x_test, y_test, epochs=5)

print('predict -------------------------------------')
a_test = model.predict(x_test)
a_test = np.round(np.maximum(np.minimum(a_test, 9), 0))

diff = [1 if (y_test[0][i] == a_test[0][i]) else 0 for i in range(len(y_test[0]))]
accuracy = 100 * np.sum(diff) / len(diff)

print('accuracy:', accuracy, '%')
