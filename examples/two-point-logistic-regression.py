import numpy as np
import os, sys
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir + '/../')
import library as nn

x = np.array([1, 2])
y = np.array([0, 1])

print('train------------------------------------------------')

model = nn.models.Sequential(lr = 0.2, loss = 'log')
model.add(nn.layers.Dense(units = 1, input_dim = 1, activation = 'sigmoid'))
model.fit(x, y, epochs = 100)

x_test = np.array([-10 ** p for p in range(10, 0, -1)] + [0 , 1, 2, 3] + [10 ** p for p in range(1, 11)])
y_test = np.array([0 if v <= 1 else 1 for v in x_test])

print('weights and biases ------------------------------------------------')
print([[layer.w, layer.b] for layer in model.layers])

print('predict------------------------------------------------')

y_predict = model.predict(x_test).reshape((len(x_test), ))
y_predict = np.floor(y_predict * 1.9999)
y_loss = [(y_predict[i] - y_test[i]) / (y_test[i] or 1) for i in range(len(y_test))]

print(np.array(list(zip(x_test, y_test, y_predict, y_loss))))

diff = [1 if (y_test[i] == y_predict[i]) else 0 for i in range(len(y_test))]
accuracy_rate = 100 * np.sum(diff) / len(diff)

print('accuracy_rate:', accuracy_rate, '% ===========================')
