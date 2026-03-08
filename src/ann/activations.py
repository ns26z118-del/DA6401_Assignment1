#activations.py
import numpy as np

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, z):
        """
        Forward pass: f(z) = 1 / (1 + e^(-z))
        """
        z = np.clip(z, -500, 500) # Prevent overflow
        self.out = 1.0 / (1.0 + np.exp(-z))
        return self.out

    def backward(self, d_out):
        """
        Backward pass: local derivative * incoming gradient
        """
        local_grad = self.out * (1.0 - self.out)
        return d_out * local_grad

class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, z):
        """
        Forward pass: f(z) = tanh(z)
        """
        self.out = np.tanh(z)
        return self.out

    def backward(self, d_out):
        """
        Backward pass: local derivative * incoming gradient
        """
        local_grad = 1.0 - np.power(self.out, 2)
        return d_out * local_grad

class ReLU:
    def __init__(self):
        self.z = None

    def forward(self, z):
        """
        Forward pass: f(z) = max(0, z)
        """
        self.z = z
        return np.maximum(0, z)

    def backward(self, d_out):
        """
        Backward pass: local derivative * incoming gradient
        """
        local_grad = (self.z > 0).astype(float)
        return d_out * local_grad