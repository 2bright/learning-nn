import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import library as nn

mnist = nn.datasets.mnist

(x_train, x_test) = mnist.load_images(column_vector=True, normalized=True)
(y_train, y_test) = mnist.load_labels(column_vector=True, bit_vector=True)

model = nn.models.Sequential(layers = [
  nn.layers.Dense(units = 1, input_dim = 1, activation = 'relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
], lr = 0.2, loss = 'log')

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
