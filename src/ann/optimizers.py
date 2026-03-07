
import numpy as np


class SGD:
    def __init__(self, lr, weight_decay=0.0):
        self.lr = lr
        self.wd = weight_decay

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "W"):

                grad_W = layer.grad_W + self.wd * layer.W
                grad_b = layer.grad_b

                layer.W -= self.lr * grad_W
                layer.b -= self.lr * grad_b


class Momentum:
    def __init__(self, lr, weight_decay=0.0, beta=0.9):
        self.lr = lr
        self.wd = weight_decay
        self.beta = beta

        self.v_W = {}
        self.v_b = {}

    def step(self, layers):

        for i, layer in enumerate(layers):

            if hasattr(layer, "W"):

                grad_W = layer.grad_W + self.wd * layer.W
                grad_b = layer.grad_b

                if i not in self.v_W:
                    self.v_W[i] = np.zeros_like(layer.W)
                    self.v_b[i] = np.zeros_like(layer.b)

                self.v_W[i] = self.beta * self.v_W[i] + (1 - self.beta) * grad_W
                self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * grad_b

                layer.W -= self.lr * self.v_W[i]
                layer.b -= self.lr * self.v_b[i]


class NAG:
    def __init__(self, lr, weight_decay=0.0, beta=0.9):
        self.lr = lr
        self.wd = weight_decay
        self.beta = beta

        self.v_W = {}
        self.v_b = {}

    def step(self, layers):

        for i, layer in enumerate(layers):

            if hasattr(layer, "W"):

                grad_W = layer.grad_W + self.wd * layer.W
                grad_b = layer.grad_b

                if i not in self.v_W:
                    self.v_W[i] = np.zeros_like(layer.W)
                    self.v_b[i] = np.zeros_like(layer.b)

                v_prev_W = self.v_W[i].copy()
                v_prev_b = self.v_b[i].copy()

                self.v_W[i] = self.beta * self.v_W[i] + self.lr * grad_W
                self.v_b[i] = self.beta * self.v_b[i] + self.lr * grad_b

                layer.W -= (-self.beta * v_prev_W + (1 + self.beta) * self.v_W[i])
                layer.b -= (-self.beta * v_prev_b + (1 + self.beta) * self.v_b[i])


class RMSProp:
    def __init__(self, lr, weight_decay=0.0, beta=0.9, eps=1e-8):
        self.lr = lr
        self.wd = weight_decay
        self.beta = beta
        self.eps = eps

        self.s_W = {}
        self.s_b = {}

    def step(self, layers):

        for i, layer in enumerate(layers):

            if hasattr(layer, "W"):

                grad_W = layer.grad_W + self.wd * layer.W
                grad_b = layer.grad_b

                if i not in self.s_W:
                    self.s_W[i] = np.zeros_like(layer.W)
                    self.s_b[i] = np.zeros_like(layer.b)

                self.s_W[i] = self.beta * self.s_W[i] + (1 - self.beta) * (grad_W ** 2)
                self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * (grad_b ** 2)

                layer.W -= self.lr * grad_W / (np.sqrt(self.s_W[i]) + self.eps)
                layer.b -= self.lr * grad_b / (np.sqrt(self.s_b[i]) + self.eps)