import numpy as np
import os, sys
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir + '/../')
import library as nn

digit = int(sys.argv[1]) if len(sys.argv) >= 2 else 0

print('test digit:', digit)

mnist = nn.datasets.mnist

(x_train, y_train, x_test, y_test) = mnist.load(file_dir + '/../data/mnist')

y_train = y_train[digit:digit + 1, :]
y_test = y_test[digit:digit + 1, :]

print('train -------------------------------------')
model = nn.models.Sequential(lr = 0.1, loss = 'log', batch_size = 50, layers = [
    nn.layers.Dense(units = y_train.shape[0], input_dim = x_train.shape[0], activation = 'sigmoid')
], optimizer = 'gd')

model.fit(x_train, y_train, epochs=20)

print('metrics for train set -------------------------------------')
a_test = model.predict(x_train)
a_test = np.floor(a_test * 1.9999)

diff_0 = [1 if (y_test[0][i] == 0 and a_test[0][i] == 0) else 0 for i in range(len(y_test[0]))]
diff_1 = [1 if (y_test[0][i] == 1 and a_test[0][i] == 1) else 0 for i in range(len(y_test[0]))]

accuracy_0 = 100 * np.sum(diff_0) / - np.sum(y_test - 1)
accuracy_1 = 100 * np.sum(diff_1) / np.sum(y_test)

print('accuracy_0:', accuracy_0, '%')
print('accuracy_1:', accuracy_1, '%')

print('metrics for test set -------------------------------------')
a_test = model.predict(x_test)
a_test = np.floor(a_test * 1.9999)

diff_0 = [1 if (y_test[0][i] == 0 and a_test[0][i] == 0) else 0 for i in range(len(y_test[0]))]
diff_1 = [1 if (y_test[0][i] == 1 and a_test[0][i] == 1) else 0 for i in range(len(y_test[0]))]

accuracy_0 = 100 * np.sum(diff_0) / - np.sum(y_test - 1)
accuracy_1 = 100 * np.sum(diff_1) / np.sum(y_test)

print('accuracy_0:', accuracy_0, '%')
print('accuracy_1:', accuracy_1, '%')
