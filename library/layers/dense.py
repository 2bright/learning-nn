import numpy as np
import warnings

warnings.filterwarnings('ignore', message = 'overflow encountered in exp')

class Dense:
    def __init__(self, units, input_dim, activation):
        self.w = np.random.randn(units, input_dim) * 0.01
        self.b = np.zeros((units, 1))
        self.activation = activation

    def a(self, z):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        else: # self.activation == 'linear'
            return z

    def d_a(self, z, a):
        if self.activation == 'sigmoid':
            return a * (1 - a)
        else: # self.activation == 'linear'
            return 1
