import numpy as np

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

        for itr in range(epochs):
            y_hat = np.ndarray((y.shape[0], 0))
            for batch in range(B):
                cache_z = [None] * L
                cache_a = [None] * (L + 1)
                cache_a[0] = x[:, batch * self.batch_size:(batch + 1) * self.batch_size]
                _batch_size = cache_a[0].shape[1]

                for l in range(L):
                    layer = self.layers[l]
                    cache_z[l] = np.dot(layer.w, cache_a[l]) + layer.b
                    cache_a[l + 1] = layer.a(cache_z[l])

                y_hat = np.concatenate((y_hat, cache_a[L]), axis = 1)
                cache_da = self.d_loss(cache_a[L], y[:, batch * self.batch_size:(batch + 1) * self.batch_size])

                for l in reversed(range(L)):
                    layer = self.layers[l]

                    cache_dz = cache_da * layer.d_a(cache_z[l], cache_a[l+1])
                    cache_da = np.dot(layer.w.transpose(), cache_dz)

                    dw = np.dot(cache_dz, cache_a[l].transpose()) / _batch_size
                    db = np.sum(cache_dz, axis = 1, keepdims = True) / _batch_size

                    layer.w -= self.lr * dw
                    layer.b -= self.lr * db

            print("iteration %10d, error %.6f"% (itr + 1, self.error(y_hat, y)))

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

    def error(self, a, y):
        batch_size = y.shape[1]
        if self.loss == 'log':
            return np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a)) / batch_size
        else:
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
