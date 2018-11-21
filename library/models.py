import numpy as np
import time
import warnings
warnings.filterwarnings('ignore', category = RuntimeWarning)

class Sequential:
    def __init__(self, lr, loss, batch_size = 100, layers=[]):
        self.lr = lr
        self.loss = loss
        self.layers = layers
        self.batch_size = batch_size

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, x, y, epochs):
        if x.ndim == 1:
            x = x.reshape(1, len(x))

        if y.ndim == 1:
            y = y.reshape(1, len(y))

        if x.ndim != 2 or y.ndim != 2:
            raise Exception('x y argument of fit should be both 2d numpy array.')

        L = len(self.layers)
        B = int(np.ceil(x.shape[1] / self.batch_size))
        time_progress = time.time()

        for itr in range(epochs):
            for batch in range(B):
                cache_z = [None] * L
                cache_a = [None] * (L + 1)
                cache_a[0] = x[:, batch * self.batch_size:(batch + 1) * self.batch_size]
                _batch_y = y[:, batch * self.batch_size:(batch + 1) * self.batch_size]
                _batch_size = cache_a[0].shape[1]

                for l in range(L):
                    layer = self.layers[l]
                    cache_z[l] = np.dot(layer.w, cache_a[l]) + layer.b
                    cache_a[l + 1] = layer.a(cache_z[l])

                if time.time() - time_progress > 0.5:
                    time_progress = time.time()
                    print("\riteration %10d, mean_loss: %.6f"% (itr + 1, self.mean_loss(cache_a[L], _batch_y)), end='')

                cache_da = self.d_loss(cache_a[L], _batch_y)

                for l in reversed(range(L)):
                    layer = self.layers[l]

                    cache_dz = cache_da * layer.d_a(cache_z[l], cache_a[l+1])

                    dw = np.dot(cache_dz, cache_a[l].transpose()) / _batch_size
                    db = np.sum(cache_dz, axis = 1, keepdims = True) / _batch_size

                    if layer.activation == 'x_relu':
                        dt = layer.dt(cache_da, cache_z[l]) / _batch_size
                        dp = layer.dp(cache_da, cache_z[l]) / _batch_size
                        dn = layer.dn(cache_da, cache_z[l]) / _batch_size

                        layer.x_relu_t -= self.lr * dt
                        layer.x_relu_p -= self.lr * dp
                        layer.x_relu_n -= self.lr * dn

                    cache_da = np.dot(layer.w.transpose(), cache_dz)

                    layer.w -= self.lr * dw
                    layer.b -= self.lr * db

            print("\riteration %10d, mean_loss: %.6f"% (itr + 1, self.mean_loss(cache_a[L], _batch_y)))

    def d_loss(self, a, y):
        if self.loss == 'log':
            def d_log_loss(_a, _y):
                if (_y == 1 and _a == 0) or (_y == 0 and _a == 1):
                    return -10000000000.0
                elif (_y == 1 and _a == 1) or (_y == 0 and _a == 0):
                    return -1.0
                else:
                    return -_y/_a + (1 - _y)/(1 - _a)
            return np.vectorize(d_log_loss)(a, y)
        else: # self.loss == 'se'
            return 2 * (a - y)

    def mean_loss(self, a, y):
        batch_size = y.shape[1]
        if self.loss == 'log':
            return np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a)) / batch_size
        else: # self.loss == 'se'
            return np.sum(np.abs(a - y)) / batch_size

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, len(x))

        if x.ndim != 2:
            raise Exception('x argument of predict should be a 2d numpy array.')

        L = len(self.layers)
        a = x

        for l in range(L):
            layer = self.layers[l]
            z = np.dot(layer.w, a) + layer.b
            a = layer.a(z)

        return a
