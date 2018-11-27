import numpy as np

class Dense:
    def __init__(self, units, input_dim, activation):
        self.w = np.random.randn(units, input_dim) * 0.01
        self.b = np.zeros((units, 1))
        self.x_relu_t = np.zeros((units, 1))
        self.x_relu_p = np.ones((units, 1))
        self.x_relu_n = np.zeros((units, 1))
        self.activation = activation

    def a(self, z):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'x_relu':
            def x_relu(v, t, p, n):
                return (p if v > t else n) * (v - t) + t
            return np.vectorize(x_relu)(z, self.x_relu_t, self.x_relu_p, self.x_relu_n)
        else: # self.activation == 'linear'
            return z

    def d_a(self, z, a):
        if self.activation == 'sigmoid':
            return a * (1 - a)
        elif self.activation == 'relu':
            def d_relu(z):
                return 1 if z > 0 else 0
            return np.vectorize(d_relu)(z)
        elif self.activation == 'x_relu':
            def d_x_relu(v, t, p, n):
                return p if v > t else n
            return np.vectorize(d_x_relu)(z, self.x_relu_t, self.x_relu_p, self.x_relu_n)
        else: # self.activation == 'linear'
            return 1

    def dt(self, da, z):
        def _dt(_z, t, p, n):
            return (1 - p) if _z > t else (1 - n)
        return np.einsum('ij,ij->i', da, np.vectorize(_dt)(z, self.x_relu_t, self.x_relu_p, self.x_relu_n)).reshape(da.shape[0], 1)

    def dp(self, da, z):
        def _dp(_z, t):
            return (_z - t) if _z > t else 0
        return np.einsum('ij,ij->i', da, np.vectorize(_dp)(z, self.x_relu_t)).reshape(da.shape[0], 1)

    def dn(self, da, z):
        def _dn(_z, t):
            return (_z - t) if _z <= t else 0
        return np.einsum('ij,ij->i', da, np.vectorize(_dn)(z, self.x_relu_t)).reshape(da.shape[0], 1)
