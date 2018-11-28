import numpy as np
import time
from datetime import datetime
import json
from .layers import *

class Sequential:
    def __init__(self, lr, loss, batch_size = 100, layers = [], optimizer = 'gd', save_path = './storage', L2_lambd = 0, use_batch_norm = False, epsilon = 1e-8, beta1 = 0.9, beta2 = 0.999):
        self.lr = lr
        self.loss = loss
        self.layers = layers
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.save_path = save_path
        self.L2_lambd = L2_lambd
        self.use_batch_norm = use_batch_norm
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def add(self, layer):
        self.layers.append(layer)

    def hyperparameters():
        return {
                'learning_rate': self.learning_rate,
                'loss_function': self.loss,
                'batch_size': self.batch_size,
                'layers': [layer.hyperparameters() for layer in self.layers]
                }

    def save_model(save_path = None):
        if save_path == None:
            save_path = self.save_path + '/' + datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        with open(save_path + '/hyperparameters.json') as json_file:
            json_file.write(json.dumps(self.hyperparameters()))

        with open(save_path + '/parameters.json') as json_file:
            json_file.write(json.dumps(self.parameters()))

        with open(save_path + '/metrics.json') as json_file:
            json_file.write(json.dumps(self.metrics()))

        self.save_metrics_plot(save_path)

    def save_metrics_plot(save_path = None):
        metrics = self.metrics()

        # costs through iterations


        # user specified metrics
        pass

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

        if self.optimizer == 'adam':
            adam_t = 1
            Vdg = [0] * (L + 1)
            Vdw = [0] * (L + 1)
            Vdb = [0] * (L + 1)
            Sdg = [0] * (L + 1)
            Sdw = [0] * (L + 1)
            Sdb = [0] * (L + 1)

        for itr in range(epochs):
            for batch in range(B):
                cache_z = [None] * (L + 1)
                cache_a = [None] * (L + 1)
                cache_a[0] = x[:, batch * self.batch_size:(batch + 1) * self.batch_size]
                _batch_y = y[:, batch * self.batch_size:(batch + 1) * self.batch_size]
                _B = cache_a[0].shape[1]

                if self.use_batch_norm:
                    cache_Z_bn = [None] * (L + 1)
                    cache_Z_norm = [None] * (L + 1)
                    cache_Z_svar = [None] * (L + 1)

                for l in range(1, L + 1):
                    layer = self.layers[l - 1]

                    if isinstance(layer, Dropout):
                        cache_z[l] = None
                        cache_a[l] = np.multiply(cache_a[l - 1], np.random.random(cache_a[l - 1].shape) > layer.rate) / (1 - layer.rate)
                    elif self.use_batch_norm:
                        cache_z[l] = np.dot(layer.w, cache_a[l - 1])

                        mean = np.mean(cache_z[l])
                        var = np.var(cache_z[l])

                        cache_Z_svar[l] = np.sqrt(var + self.epsilon)
                        cache_Z_norm[l] = (cache_z[l] - mean) / cache_Z_svar[l]
                        cache_Z_bn[l] = np.multiply(layer.g, cache_Z_norm[l]) + layer.b
                        cache_a[l] = layer.a(cache_Z_bn[l])
                    else:
                        cache_z[l] = np.dot(layer.w, cache_a[l - 1]) + layer.b
                        #print('layer:', l, 'max z:', np.max(cache_z[l]), 'max w:', np.max(layer.w), 'max b:', np.max(layer.b), 'max a:', np.max(cache_a[l - 1]))
                        #print('layer:', l, 'min z:', np.min(cache_z[l]), 'min w:', np.min(layer.w), 'min b:', np.min(layer.b), 'min a:', np.min(cache_a[l - 1]))
                        cache_a[l] = layer.a(cache_z[l])

                if time.time() - time_progress > 0.5:
                    time_progress = time.time()
                    print("\riteration %10d, mean_loss: %.6f"% (itr + 1, self.mean_loss(cache_a[L], _batch_y)), end='')

                cache_da = self.d_loss(cache_a[L], _batch_y)

                for l in reversed(range(1, L + 1)):
                    layer = self.layers[l - 1]

                    if isinstance(layer, Dropout):
                        continue

                    if self.use_batch_norm:
                        cache_dz_bn = layer.d_a(cache_Z_bn[l], cache_a[l], cache_da)
                        cache_dz = cache_dz_bn * layer.g * (1 - 1 / _B - (cache_Z_bn[l] * cache_Z_bn[l]) / _B) / cache_Z_svar[l]

                        dG = np.einsum('ij,ij->i', cache_dz_bn, cache_Z_norm[l]).reshape(cache_dz_bn.shape[0], 1) / _B
                        dW = np.dot(cache_dz, cache_a[l - 1].T) / _B
                        dB = np.sum(cache_dz_bn, axis = 1, keepdims = True) / _B

                        if self.optimizer == 'adam':
                            Vdg[l] = self.beta1 * Vdg[l] + (1 - self.beta1) * dG
                            Vdw[l] = self.beta1 * Vdw[l] + (1 - self.beta1) * dW
                            Vdb[l] = self.beta1 * Vdb[l] + (1 - self.beta1) * dB
                            Sdg[l] = self.beta2 * Sdg[l] + (1 - self.beta2) * dG * dG
                            Sdw[l] = self.beta2 * Sdw[l] + (1 - self.beta2) * dW * dW
                            Sdb[l] = self.beta2 * Sdb[l] + (1 - self.beta2) * dB * dB
                    else:
                        cache_dz = layer.d_a(cache_z[l], cache_a[l], cache_da)

                        dW = np.dot(cache_dz, cache_a[l - 1].T) / _B
                        dB = np.sum(cache_dz, axis = 1, keepdims = True) / _B

                        if self.optimizer == 'adam':
                            Vdw[l] = self.beta1 * Vdw[l] + (1 - self.beta1) * dW
                            Vdb[l] = self.beta1 * Vdb[l] + (1 - self.beta1) * dB
                            Sdw[l] = self.beta2 * Sdw[l] + (1 - self.beta2) * dW * dW
                            Sdb[l] = self.beta2 * Sdb[l] + (1 - self.beta2) * dB * dB

                    if callable(self.lr):
                        lr = self.lr(itr)
                    else:
                        lr = self.lr

                    if layer.activation == 'x_relu':
                        # TODO adam
                        dt = layer.dt(cache_da, cache_z[l]) / _B
                        dp = layer.dp(cache_da, cache_z[l]) / _B
                        dn = layer.dn(cache_da, cache_z[l]) / _B

                        layer.x_relu_t -= lr * dt
                        layer.x_relu_p -= lr * dp
                        layer.x_relu_n -= lr * dn

                    cache_da = np.dot(layer.w.T, cache_dz)

                    if self.optimizer == 'adam':
                        V_fix = 1 - np.power(self.beta1, adam_t)
                        S_fix = 1 - np.power(self.beta2, adam_t)
                        adam_t += 1
                        if self.use_batch_norm:
                            layer.g -= lr * (Vdg[l] / V_fix) / (np.sqrt(Sdg[l] / S_fix) + self.epsilon)
                            layer.w = (1 - lr * self.L2_lambd / _B) * layer.w - lr * (Vdw[l] / V_fix) / (np.sqrt(Sdw[l] / S_fix) + self.epsilon)
                            layer.b -= lr * (Vdb[l] / V_fix) / (np.sqrt(Sdb[l] / S_fix) + self.epsilon)
                        else:
                            _dW = (Vdw[l] / V_fix) / (np.sqrt(Sdw[l] / S_fix) + self.epsilon)
                            _dB = (Vdb[l] / V_fix) / (np.sqrt(Sdb[l] / S_fix) + self.epsilon)

                            #max_dw = np.amax(np.absolute(_dW))
                            #max_db = np.amax(np.absolute(_dB))

                            #print('---------------', max_dw, max_db)

                            #if max_dw > 1:
                            #    _dW /= max_dw

                            #if max_db > 1:
                            #    _dB /= max_db

                            #print('---------------', np.amax(np.absolute(_dW)), np.amax(np.absolute(_dB)))

                            layer.w -= lr * _dW
                            layer.b -= lr * _dB
                    else:
                        if self.use_batch_norm:
                            layer.g -= lr * dG
                            layer.w = (1 - lr * self.L2_lambd / _B) * layer.w - lr * dW
                            layer.b -= lr * dB
                        else:
                            layer.w -= lr * dW
                            layer.b -= lr * dB

            print("\riteration %10d, mean_loss: %.6f"% (itr + 1, self.mean_loss(cache_a[L], _batch_y)))

    def d_loss(self, a, y):
        if self.loss == 'log':
            def d_log_loss(_a, _y):
                if (_y == 1 and _a == 0) or (_y == 0 and _a == 1):
                    return -10.0
                elif (_y == 1 and _a == 1) or (_y == 0 and _a == 0):
                    return -1.0
                else:
                    return -_y/_a + (1 - _y)/(1 - _a)
            return np.vectorize(d_log_loss)(a, y)
        elif self.loss == 'softmax_log':
            def d_softmax_log_loss(_a, _y):
                if _y == 1 and _a == 0:
                    return -10.0
                elif _y == 0 and _a == 0:
                    return 0.0
                else:
                    return -_y / _a
            return np.vectorize(d_softmax_log_loss)(a, y)
        else: # self.loss == 'se'
            return 2 * (a - y)

    def mean_loss(self, a, y):
        batch_size = y.shape[1]
        if self.loss == 'log':
            a = a * (1 - 1e-8) + 5e-9
            per_loss = -y * np.log(a) - (1 - y) * np.log(1 - a)
            res = np.sum(per_loss) / batch_size
            return res
        elif self.loss == 'softmax_log':
            a = a * (1 - 1e-8) + 5e-9
            per_loss = -y * np.log(a)
            res = np.sum(per_loss) / batch_size
            return res
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

            if isinstance(layer, Dropout):
                continue

            z = np.dot(layer.w, a) + layer.b
            a = layer.a(z)

        return a
