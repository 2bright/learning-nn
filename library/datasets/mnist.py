import numpy as np
import struct
from array import array

train_images_filename = 'train-images-idx3-ubyte'
train_labels_filename = 'train-labels-idx1-ubyte'
test_images_filename = 't10k-images-idx3-ubyte'
test_labels_filename = 't10k-labels-idx1-ubyte'

def load(mnist_dir):
    return _load(mnist_dir + '/' + train_images_filename), \
        _load(mnist_dir + '/' + train_labels_filename), \
        _load(mnist_dir + '/' + test_images_filename), \
        _load(mnist_dir + '/' + test_labels_filename)

def _load(file_path):
    with open(file_path, 'rb') as file:
        magic, size = struct.unpack('>II', file.read(8))

        if magic != 2051 and magic != 2049:
            raise Exception('mnist magic bumber error: ', 'got', magic, file_path)

        if magic == 2051: # images
            rows, cols = struct.unpack('>II', file.read(8))
            vector_size = rows * cols
        else: # labels
            vector_size = 10

        mnist_data = array("B", file.read())

    data = np.zeros((size, vector_size))

    for i in range(size):
        if magic == 2049:
            data[i][mnist_data[i]] = 1
        else:
            data[i] = mnist_data[i * vector_size:(i + 1) * vector_size]

    if magic == 2051:
        data /= 255.0

    return data.transpose()

def b2d_element_wise(v):
    max_e = 0.5
    d = 0

    for i in range(10):
        if v[i] > max_e:
            max_e = v[i]
            d = i

    return d if max_e > 0.5 else -1

def b2d(y):
    return np.array([[b2d_element_wise(v) for v in y.transpose()]])

def preview(indices, x, y):
    x = x.transpose()
    y_d = b2d(y)
    for i in indices:
        p  = x[i]
        print('---------------------------', y_d[0][i], '----------------------------')
        for h in range(28):
            print('|', end='')
            for w in range(28):
                print('  ' if p[h * 28 + w] == 0 else '[]', end='')
            print('|')
        print('----------------------------------------------------------')
