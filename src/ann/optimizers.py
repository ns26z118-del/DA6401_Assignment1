"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp, Adam, Nadam
"""
#optimizers.py
import numpy as np

class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.wd = weight_decay

    def update(self, layers):
        """SGD: simple gradient descent."""
        for layer in layers:
            if hasattr(layer, 'W'):
                grad_W = layer.grad_W + self.wd * layer.W
                grad_b = layer.grad_b

                layer.W -= self.lr * grad_W
                layer.b -= self.lr * grad_b


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.wd = weight_decay
        self.velocities = {}

    def update(self, layers):
        for layer in layers:
            if hasattr(layer, 'W'):
                layer_id = id(layer)
                if layer_id not in self.velocities:
                    self.velocities[layer_id] = {
                        'W': np.zeros_like(layer.W),
                        'b': np.zeros_like(layer.b)
                    }

                grad_W = layer.grad_W + self.wd * layer.W
                grad_b = layer.grad_b

                self.velocities[layer_id]['W'] = (self.momentum * self.velocities[layer_id]['W']
                                                   + self.lr * grad_W)
                self.velocities[layer_id]['b'] = (self.momentum * self.velocities[layer_id]['b']
                                                   + self.lr * grad_b)

                layer.W -= self.velocities[layer_id]['W']
                layer.b -= self.velocities[layer_id]['b']


class NAG:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.wd = weight_decay
        self.velocities = {}

    def update(self, layers):
        """Nesterov Accelerated Gradient — corrected formula."""
        for layer in layers:
            if hasattr(layer, 'W'):
                layer_id = id(layer)
                if layer_id not in self.velocities:
                    self.velocities[layer_id] = {
                        'W': np.zeros_like(layer.W),
                        'b': np.zeros_like(layer.b)
                    }

                grad_W = layer.grad_W + self.wd * layer.W
                grad_b = layer.grad_b

                v_prev_W = self.velocities[layer_id]['W'].copy()
                v_prev_b = self.velocities[layer_id]['b'].copy()

                # Update velocity
                self.velocities[layer_id]['W'] = (self.momentum * v_prev_W + self.lr * grad_W)
                self.velocities[layer_id]['b'] = (self.momentum * v_prev_b + self.lr * grad_b)

                # Nesterov update: use lookahead velocity, not double-momentum
                layer.W -= (self.momentum * self.velocities[layer_id]['W'] + self.lr * grad_W)
                layer.b -= (self.momentum * self.velocities[layer_id]['b'] + self.lr * grad_b)


class RMSProp:
    def __init__(self, lr=0.001, beta=0.99, epsilon=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.wd = weight_decay
        self.s = {}

    def update(self, layers):
        for layer in layers:
            if hasattr(layer, 'W'):
                layer_id = id(layer)
                if layer_id not in self.s:
                    self.s[layer_id] = {
                        'W': np.zeros_like(layer.W),
                        'b': np.zeros_like(layer.b)
                    }

                grad_W = layer.grad_W + self.wd * layer.W
                grad_b = layer.grad_b

                self.s[layer_id]['W'] = (self.beta * self.s[layer_id]['W']
                                         + (1 - self.beta) * np.square(grad_W))
                self.s[layer_id]['b'] = (self.beta * self.s[layer_id]['b']
                                         + (1 - self.beta) * np.square(grad_b))

                layer.W -= (self.lr / (np.sqrt(self.s[layer_id]['W']) + self.epsilon)) * grad_W
                layer.b -= (self.lr / (np.sqrt(self.s[layer_id]['b']) + self.epsilon)) * grad_b


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.wd = weight_decay
        self.m = {}   # First moment (mean)
        self.v = {}   # Second moment (variance)
        self.t = {}   # Timestep per layer

    def update(self, layers):
        """Adam: Adaptive Moment Estimation."""
        for layer in layers:
            if hasattr(layer, 'W'):
                layer_id = id(layer)
                if layer_id not in self.m:
                    self.m[layer_id] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                    self.v[layer_id] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                    self.t[layer_id] = 0

                self.t[layer_id] += 1
                t = self.t[layer_id]

                grad_W = layer.grad_W + self.wd * layer.W
                grad_b = layer.grad_b

                # Update biased first & second moment estimates
                self.m[layer_id]['W'] = self.beta1 * self.m[layer_id]['W'] + (1 - self.beta1) * grad_W
                self.m[layer_id]['b'] = self.beta1 * self.m[layer_id]['b'] + (1 - self.beta1) * grad_b

                self.v[layer_id]['W'] = self.beta2 * self.v[layer_id]['W'] + (1 - self.beta2) * np.square(grad_W)
                self.v[layer_id]['b'] = self.beta2 * self.v[layer_id]['b'] + (1 - self.beta2) * np.square(grad_b)

                # Bias-corrected estimates
                m_hat_W = self.m[layer_id]['W'] / (1 - self.beta1 ** t)
                m_hat_b = self.m[layer_id]['b'] / (1 - self.beta1 ** t)
                v_hat_W = self.v[layer_id]['W'] / (1 - self.beta2 ** t)
                v_hat_b = self.v[layer_id]['b'] / (1 - self.beta2 ** t)

                layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
                layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)


class Nadam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.wd = weight_decay
        self.m = {}
        self.v = {}
        self.t = {}

    def update(self, layers):
        """Nadam: Adam with Nesterov momentum."""
        for layer in layers:
            if hasattr(layer, 'W'):
                layer_id = id(layer)
                if layer_id not in self.m:
                    self.m[layer_id] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                    self.v[layer_id] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                    self.t[layer_id] = 0

                self.t[layer_id] += 1
                t = self.t[layer_id]

                grad_W = layer.grad_W + self.wd * layer.W
                grad_b = layer.grad_b

                self.m[layer_id]['W'] = self.beta1 * self.m[layer_id]['W'] + (1 - self.beta1) * grad_W
                self.m[layer_id]['b'] = self.beta1 * self.m[layer_id]['b'] + (1 - self.beta1) * grad_b

                self.v[layer_id]['W'] = self.beta2 * self.v[layer_id]['W'] + (1 - self.beta2) * np.square(grad_W)
                self.v[layer_id]['b'] = self.beta2 * self.v[layer_id]['b'] + (1 - self.beta2) * np.square(grad_b)

                # Bias-corrected second moment
                v_hat_W = self.v[layer_id]['W'] / (1 - self.beta2 ** t)
                v_hat_b = self.v[layer_id]['b'] / (1 - self.beta2 ** t)

                # Nadam: use Nesterov lookahead for first moment
                m_nadam_W = (self.beta1 * self.m[layer_id]['W'] / (1 - self.beta1 ** (t + 1))
                             + (1 - self.beta1) * grad_W / (1 - self.beta1 ** t))
                m_nadam_b = (self.beta1 * self.m[layer_id]['b'] / (1 - self.beta1 ** (t + 1))
                             + (1 - self.beta1) * grad_b / (1 - self.beta1 ** t))

                layer.W -= self.lr * m_nadam_W / (np.sqrt(v_hat_W) + self.epsilon)
                layer.b -= self.lr * m_nadam_b / (np.sqrt(v_hat_b) + self.epsilon)
