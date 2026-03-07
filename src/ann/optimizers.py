"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np

class SGD:

    def __init__(self, lr):
        self.lr= lr

    def update(self, layer):
        layer.W -= self.lr * layer.grad_W
        layer.b -= self.lr * layer.grad_b


class Momentum:

    def __init__(self, lr, beta=0.9):
        self.lr= lr
        self.beta= beta
        self.vW= {}
        self.vb= {}

    def update(self, layer, i):
        if i not in self.vW:
            self.vW[i]= np.zeros_like(layer.W)
            self.vb[i]= np.zeros_like(layer.b)

        self.vW[i]= self.beta * self.vW[i] + layer.grad_W
        self.vb[i]= self.beta * self.vb[i] + layer.grad_b

        layer.W -= self.lr * self.vW[i]
        layer.b -= self.lr * self.vb[i]


class RMSProp:

    def __init__(self, lr, beta=0.9, eps=1e-8):
        self.lr= lr
        self.beta= beta
        self.eps= eps
        self.sW= {}
        self.sb= {}

    def update(self, layer, i):
        if i not in self.sW:
            self.sW[i]= np.zeros_like(layer.W)
            self.sb[i]= np.zeros_like(layer.b)

        self.sW[i]= self.beta * self.sW[i] + (1 - self.beta) * (layer.grad_W ** 2)
        self.sb[i]= self.beta * self.sb[i] + (1 - self.beta) * (layer.grad_b ** 2)

        layer.W -= self.lr * layer.grad_W / (np.sqrt(self.sW[i]) + self.eps)
        layer.b -= self.lr * layer.grad_b / (np.sqrt(self.sb[i]) + self.eps)