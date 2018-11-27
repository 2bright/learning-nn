import numpy as np
import tensorflow as tf

import os, sys
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir + '/../')

import library as nn

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)

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
