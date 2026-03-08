"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .neural_layer import DenseLayer
from .activations import ReLU, Sigmoid, Tanh
from .objective_functions import CrossEntropy, MeanSquaredError
from .optimizers import SGD, Momentum, NAG, RMSProp

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        def get_arg(key, default):
            if isinstance(cli_args, dict):
                return cli_args.get(key, default)
            return getattr(cli_args, key, default)

        self.input_dim = get_arg('input_dim', 784)  
        self.output_dim = get_arg('output_dim', 10)
        
        self.layers = [] 
        self.activations = [] 
        
        hidden_sizes = get_arg('hidden_size', [128, 128, 128])
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * get_arg('num_layers', 1)

        activation_name = get_arg('activation', 'tanh')
        weight_init = get_arg('weight_init', 'xavier')
        loss_name = get_arg('loss', 'cross_entropy')

        if activation_name.lower() == 'relu':
            ActivationClass = ReLU
        elif activation_name.lower() == 'sigmoid':
            ActivationClass = Sigmoid
        else:
            ActivationClass = Tanh

        if loss_name.lower() == 'mse':
            self.loss_fn = MeanSquaredError()
        else:
            self.loss_fn = CrossEntropy()

        opt_name = get_arg('optimizer', 'rmsprop').lower()
        lr = get_arg('learning_rate', 0.001)
        
        if opt_name == 'sgd':
            self.optimizer = SGD(lr=lr)
        elif opt_name == 'momentum':
            self.optimizer = Momentum(lr=lr)
        elif opt_name == 'nag':
            self.optimizer = NAG(lr=lr)
        else:
            self.optimizer = RMSProp(lr=lr)

        current_dim = self.input_dim
        for hidden_size in hidden_sizes:
            self.layers.append(DenseLayer(current_dim, hidden_size, weight_init))
            self.activations.append(ActivationClass())
            current_dim = hidden_size

        self.layers.append(DenseLayer(current_dim, self.output_dim, weight_init))

    def forward(self, X):
        out = X
        for i, layer in enumerate(self.layers):
            out = layer.forward(out)
            if i < len(self.activations):
                out = self.activations[i].forward(out)
        return out

    def backward(self, y_true, y_pred):
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            num_classes = y_pred.shape[1]
            y_oh = np.zeros((y_true.size, num_classes))
            y_oh[np.arange(y_true.size), y_true.flatten().astype(int)] = 1.0
            y_true = y_oh

        _ = self.loss_fn(y_pred, y_true) 
        d_out = self.loss_fn.derivative()

        grad_W_list = []
        grad_b_list = []

        # 1. Backprop Last Layer (No activation)
        last_layer = self.layers[-1]
        d_out = last_layer.backward(d_out)
        grad_W_list.append(last_layer.grad_W)
        grad_b_list.append(np.squeeze(last_layer.grad_b))

        # 2. Backprop Hidden Layers (Activation then Dense)
        for i in range(len(self.layers)-2, -1, -1):
            d_out = self.activations[i].backward(d_out)
            d_out = self.layers[i].backward(d_out)
            grad_W_list.append(self.layers[i].grad_W)
            grad_b_list.append(np.squeeze(self.layers[i].grad_b))

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.update(self.layers)

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        if y_train.ndim == 1 or y_train.shape[1] == 1:
            num_classes = self.output_dim
            y_oh = np.zeros((y_train.size, num_classes))
            y_oh[np.arange(y_train.size), y_train.flatten().astype(int)] = 1.0
            y_train = y_oh

        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                logits = self.forward(X_batch)
                self.backward(y_batch, logits)
                self.update_weights()

    def evaluate(self, X, y):
        logits = self.forward(X)
        preds = np.argmax(logits, axis=1)
        
        num_classes = logits.shape[1]
        y_oh = np.zeros((y.size, num_classes))
        y_oh[np.arange(y.size), y] = 1.0
        
        loss = self.loss_fn(logits, y_oh)
        acc = accuracy_score(y, preds)
        
        return {
            "logits": logits,
            "loss": loss,
            "accuracy": acc,
            "f1": f1_score(y, preds, average='macro'),
            "precision": precision_score(y, preds, average='macro', zero_division=0),
            "recall": recall_score(y, preds, average='macro', zero_division=0)
        }

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}" if f"W{i}" in weight_dict else f"layer_{i}_W"
            b_key = f"b{i}" if f"b{i}" in weight_dict else f"layer_{i}_b"
            
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
