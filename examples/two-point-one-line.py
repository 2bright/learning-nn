import numpy as np
import os, sys
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir + '/../')
import library as nn

x = np.array([1, 2])
y = x * 10

print('train------------------------------------------------')

model = nn.models.Sequential(lr = 0.2, loss = 'se')
model.add(nn.layers.Dense(units = 1, input_dim = 1, activation = 'linear'))
model.fit(x, y, epochs = 500)

x_test = np.array([-10 ** p for p in range(10, 0, -1)] + [0] + [10 ** p for p in range(1, 11)])
y_test = x_test * 10

print('weights and biases ------------------------------------------------')
print([[layer.w, layer.b] for layer in model.layers])

print('predict------------------------------------------------')

y_predict = model.predict(x_test).reshape((len(x_test), ))
y_loss = [(y_predict[i] - y_test[i]) / (y_test[i] or 1) for i in range(len(y_test))]

print(np.array(list(zip(y_test, y_predict, y_loss))))
