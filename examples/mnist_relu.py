import numpy as np
import os, sys
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir + '/../')
import library as nn

mnist = nn.datasets.mnist

(x_train, y_train, x_test, y_test) = mnist.load(file_dir + '/../data/mnist')

print('train -------------------------------------')
model = nn.models.Sequential(lr = 0.1, loss = 'log', batch_size = 50, layers = [
    nn.layers.Dense(units = 512, input_dim = x_train.shape[0], activation = 'relu'),
    nn.layers.Dropout(0.2),
    nn.layers.Dense(units = y_train.shape[0], input_dim = 512, activation = 'sigmoid')
], optimizer = 'adam')

model.fit(x_train, y_train, epochs=5)

print('metrics on train set -------------------------------------')
a_train = model.predict(x_train)
a_train = np.floor(a_train * 1.9999)

diff_0 = [[1 if (y_train[j][i] == 0 and a_train[j][i] == 0) else 0 for i in range(len(y_train[j]))] for j in range(10)]
diff_1 = [[1 if (y_train[j][i] == 1 and a_train[j][i] == 1) else 0 for i in range(len(y_train[j]))] for j in range(10)]

accuracy_0 = 100 * np.sum(diff_0, axis = 1) / - np.sum(y_train - 1, axis = 1)
accuracy_1 = 100 * np.sum(diff_1, axis = 1) / np.sum(y_train, axis = 1)

for d in range(10):
    print('digit %d, accuracy_1: %g %%, accuracy_0: %g %%'% (d, accuracy_1[d], accuracy_0[d]))

# bit vector to digit

y_train_digit = mnist.b2d(y_train)
a_train_digit = mnist.b2d(a_train)

diff = [1 if (y_train_digit[0][i] == a_train_digit[0][i]) else 0 for i in range(len(y_train_digit[0]))]
accuracy = 100 * np.sum(diff) / len(diff)

print('accuracy:', accuracy, '%')

print('metrics on test set -------------------------------------')
a_test = model.predict(x_test)
a_test = np.floor(a_test * 1.9999)

diff_0 = [[1 if (y_test[j][i] == 0 and a_test[j][i] == 0) else 0 for i in range(len(y_test[j]))] for j in range(10)]
diff_1 = [[1 if (y_test[j][i] == 1 and a_test[j][i] == 1) else 0 for i in range(len(y_test[j]))] for j in range(10)]

accuracy_0 = 100 * np.sum(diff_0, axis = 1) / - np.sum(y_test - 1, axis = 1)
accuracy_1 = 100 * np.sum(diff_1, axis = 1) / np.sum(y_test, axis = 1)

for d in range(10):
    print('digit %d, accuracy_1: %g %%, accuracy_0: %g %%'% (d, accuracy_1[d], accuracy_0[d]))

# bit vector to digit

y_test_digit = mnist.b2d(y_test)
a_test_digit = mnist.b2d(a_test)

diff = [1 if (y_test_digit[0][i] == a_test_digit[0][i]) else 0 for i in range(len(y_test_digit[0]))]
accuracy = 100 * np.sum(diff) / len(diff)

print('accuracy:', accuracy, '%')
