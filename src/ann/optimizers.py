"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
#optimimzers.py 
import numpy as np

class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.wd = weight_decay

    def update(self, layers):
        """sgd is simple gradient descent and processes batched inputs."""
        for layer in layers:
            if hasattr(layer, 'W'):
                # Add L2 regularization (weight decay) to the weight gradients
                grad_W = layer.grad_W + self.wd * layer.W
                grad_b = layer.grad_b # We typically do not regularize biases

                layer.W -= self.lr * grad_W
                layer.b -= self.lr * grad_b

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.wd = weight_decay
        self.velocities = {} # Store velocity for each layer

    def update(self, layers):
        for layer in layers:
            if hasattr(layer, 'W'):
                layer_id = id(layer)
                if layer_id not in self.velocities:
                    self.velocities[layer_id] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}

                grad_W = layer.grad_W + self.wd * layer.W
                grad_b = layer.grad_b

                # Update velocities
                self.velocities[layer_id]['W'] = self.momentum * self.velocities[layer_id]['W'] + self.lr * grad_W
                self.velocities[layer_id]['b'] = self.momentum * self.velocities[layer_id]['b'] + self.lr * grad_b

                # Update weights
                layer.W -= self.velocities[layer_id]['W']
                layer.b -= self.velocities[layer_id]['b']

class NAG:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.wd = weight_decay
        self.velocities = {}

    def update(self, layers):
        """Nesterov Accelerated Gradient"""
        for layer in layers:
            if hasattr(layer, 'W'):
                layer_id = id(layer)
                if layer_id not in self.velocities:
                    self.velocities[layer_id] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}

                grad_W = layer.grad_W + self.wd * layer.W
                grad_b = layer.grad_b

                v_prev_W = self.velocities[layer_id]['W']
                v_prev_b = self.velocities[layer_id]['b']

                self.velocities[layer_id]['W'] = self.momentum * self.velocities[layer_id]['W'] + self.lr * grad_W
                self.velocities[layer_id]['b'] = self.momentum * self.velocities[layer_id]['b'] + self.lr * grad_b

                # Nesterov lookahead update
                layer.W -= (self.momentum * self.velocities[layer_id]['W'] + self.lr * grad_W)
                layer.b -= (self.momentum * self.velocities[layer_id]['b'] + self.lr * grad_b)

class RMSProp:
    def __init__(self, lr=0.001, beta=0.99, epsilon=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.wd = weight_decay
        self.s = {} # Store moving average of squared gradients

    def update(self, layers):
        for layer in layers:
            if hasattr(layer, 'W'):
                layer_id = id(layer)
                if layer_id not in self.s:
                    self.s[layer_id] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}

                grad_W = layer.grad_W + self.wd * layer.W
                grad_b = layer.grad_b

                # Update moving average of squared gradients
                self.s[layer_id]['W'] = self.beta * self.s[layer_id]['W'] + (1 - self.beta) * np.square(grad_W)
                self.s[layer_id]['b'] = self.beta * self.s[layer_id]['b'] + (1 - self.beta) * np.square(grad_b)

                # Update weights
                layer.W -= (self.lr / (np.sqrt(self.s[layer_id]['W']) + self.epsilon)) * grad_W
                layer.b -= (self.lr / (np.sqrt(self.s[layer_id]['b']) + self.epsilon)) * grad_b