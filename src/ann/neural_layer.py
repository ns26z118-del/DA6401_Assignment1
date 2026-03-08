"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

class neural_layer:

    def __init__(self, n_input, n_output, activation="linear", weight_init="xavier"):
        self.activation = activation

        if weight_init == "xavier":
            limit = np.sqrt(6 / (n_input + n_output))
            self.W = np.random.uniform(-limit, limit, (n_input, n_output))
        else:
            self.W = np.random.randn(n_input, n_output) * 0.01

        self.b = np.zeros((1, n_output))

    def forward_pass(self, X):
        self.x = X
        self.z = np.dot(X, self.W) + self.b

        if self.activation == "relu":
            self.a = np.maximum(0, self.z)
        elif self.activation == "sigmoid":
            self.a = 1 / (1 + np.exp(-self.z))
        elif self.activation == "tanh":
            self.a = np.tanh(self.z)
        else:
            self.a = self.z

        return self.a

    def backward_pass(self, grad, batch_size=None):
        if batch_size is None:
            batch_size = grad.shape[0]

        if self.activation == "relu":
            grad = grad * (self.z > 0)
        elif self.activation == "sigmoid":
            grad = grad * self.a * (1 - self.a)
        elif self.activation == "tanh":
            grad = grad * (1 - self.a**2)

        self.grad_W = np.dot(self.x.T, grad) / batch_size
        self.grad_b = np.sum(grad, axis=0, keepdims=True) / batch_size

        return np.dot(grad, self.W.T)