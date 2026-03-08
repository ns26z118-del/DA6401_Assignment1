"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
from .activations import softmax

def cross_entropy_loss(logits, y):
    probs= softmax(logits)
    m= y.shape[0]
    loss= -np.log(probs[range(m), y])
    return np.mean(loss)

def mse_loss(pred, y):
    if y.ndim == 1:
        y = np.eye(pred.shape[1])[y]
    return np.mean((pred - y) ** 2)

def cross_entropy_grad(y_true, logits):

    n = logits.shape[0]
    
    shift_logits = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shift_logits)
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    
    y_true = y_true.astype(int).flatten()
    y_one_hot = np.zeros_like(probs)
    y_one_hot[np.arange(n), y_true] = 1
    
    return (probs - y_one_hot) / n

def mse_grad(y_true, logits):
    n = logits.shape[0]
    y_true = y_true.astype(int).flatten()
    y_one_hot = np.zeros_like(logits)
    y_one_hot[np.arange(n), y_true] = 1
    
    return 2 * (logits - y_one_hot) / n