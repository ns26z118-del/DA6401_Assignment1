"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
#neural_layer.py
import numpy as np

class DenseLayer:
    def __init__(self, input_dim, output_dim, weight_init='random'):
        """
        Initializes the dense layer.
        weight_init: 'random' or 'xavier'
        """
        # Weight Initialization Strategy
        if weight_init == 'xavier':
            # Xavier/Glorot initialization: variance = 2 / (fan_in + fan_out)
            limit = np.sqrt(2.0 / (input_dim + output_dim))
            self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        else:
            # Standard random initialization scaled down
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        self.b = np.zeros((1, output_dim))

        # Mandatory variables required by the autograder
        self.grad_W = None
        self.grad_b = None

        # Cache for the forward pass input, needed for backprop
        self.X_input = None

    def forward(self, X):
        """
        Computes the forward pass: Z = X * W + b
        X shape: (batch_size, input_dim)
        """
        self.X_input = X
        # Compute the linear combination
        Z = np.dot(X, self.W) + self.b
        return Z

    def backward(self, dZ):
        """
        Computes the backward pass.
        dZ: Gradient of the loss with respect to the output Z.
        Returns dX to be passed to the previous layer.
        """
        # 1. Compute gradients for weights and biases
        # Note: We assume dZ is already averaged over the batch size by the loss function
        self.grad_W = np.dot(self.X_input.T, dZ)
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)

        # 2. Compute gradient with respect to the input X
        dX = np.dot(dZ, self.W.T)

        return dX