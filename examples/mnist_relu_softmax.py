import numpy as np
import os, sys
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir + '/../')
import library as nn

mnist = nn.datasets.mnist

(x_train, y_train, x_test, y_test) = mnist.load(file_dir + '/../data/mnist')

print('train -------------------------------------')
model = nn.models.Sequential(lr = 0.5, loss = 'softmax_log', batch_size = 50, layers = [
    nn.layers.Dense(units = 512, input_dim = x_train.shape[0], activation = 'relu'),
    nn.layers.Dropout(0.2),
    nn.layers.Dense(units = y_train.shape[0], input_dim = 512, activation = 'softmax')
], use_batch_norm = True)

model.fit(x_train, y_train, epochs=5)

print('metrics on train set -------------------------------------')
a_train = model.predict(x_train)

y_train_digit = y_train.reshape(1, len(y_train))
a_train_digit = nn.datasets.mnist.b2d_softmax(a_train.T)

diff = [1 if (y_train_digit[0][i] == a_train_digit[0][i]) else 0 for i in range(len(y_train_digit[0]))]
accuracy = 100 * np.sum(diff) / len(diff)

print('accuracy:', accuracy, '%')

print('metrics on test set -------------------------------------')
a_test = model.predict(x_test)

y_test_digit = y_test.reshape(1, len(y_test))
a_test_digit = nn.datasets.mnist.b2d_softmax(a_test.T)

diff = [1 if (y_test_digit[0][i] == a_test_digit[0][i]) else 0 for i in range(len(y_test_digit[0]))]
accuracy = 100 * np.sum(diff) / len(diff)

print('accuracy:', accuracy, '%')
