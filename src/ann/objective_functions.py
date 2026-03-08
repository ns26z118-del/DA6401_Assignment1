"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

#objective functionss.py
import numpy as np

class CrossEntropy:
    def __init__(self):
        self.softmax_out = None
        self.y_true = None

    def __call__(self, logits, y_true):
        """
        Computes Softmax then Cross-Entropy Loss.
        logits: (batch_size, num_classes)
        y_true: one-hot encoded labels (batch_size, num_classes)
        """
        # Numerically stable softmax (subtracting max prevents overflow in exp)
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted_logits)
        self.softmax_out = exps / np.sum(exps, axis=1, keepdims=True)
        self.y_true = y_true

        # Clip probabilities to prevent log(0) errors
        preds = np.clip(self.softmax_out, 1e-15, 1 - 1e-15)

        # Cross-Entropy formula over the batch
        batch_size = logits.shape[0]
        loss = -np.sum(y_true * np.log(preds)) / batch_size
        return loss

    def derivative(self):
        """
        Calculates the gradient of the loss with respect to the raw logits.
        """
        batch_size = self.y_true.shape[0]
        return (self.softmax_out - self.y_true) / batch_size


class MeanSquaredError:
    def __init__(self):
        self.logits = None
        self.y_true = None

    def __call__(self, logits, y_true):
        """
        Computes MSE directly on logits vs one-hot labels.
        """
        self.logits = logits
        self.y_true = y_true
        batch_size = logits.shape[0]

        # MSE formula
        loss = np.sum((logits - y_true) ** 2) / batch_size
        return loss

    def derivative(self):
        """
        Gradient of MSE wrt logits.
        """
        batch_size = self.y_true.shape[0]
        return 2 * (self.logits - self.y_true) / batch_size