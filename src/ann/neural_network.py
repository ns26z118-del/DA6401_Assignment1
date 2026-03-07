"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import math
import numpy as np
import wandb
from ann.neural_layer import NeuralLayer
from ann.activations import Activation
from ann.objective_functions import MSELoss, CrossEntropyLoss
from ann.optimizers import SGD, Momentum, NAG, RMSProp#, Adam, Nadam

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.cli_args = cli_args
        self.layers = []

        # Fixed for MNIST / Fashion-MNIST
        input_dim = 784
        num_classes = 10

        # -------- Build Network --------
        prev_dim = input_dim

        for hidden_dim in cli_args.hidden_size:
            self.layers.append(
                NeuralLayer(prev_dim, hidden_dim, cli_args.weight_init)
            )
            self.layers.append(
                Activation(cli_args.activation)
            )
            prev_dim = hidden_dim

        # Output layer (NO activation here)
        self.layers.append(
            NeuralLayer(prev_dim, num_classes, cli_args.weight_init)
        )

        # -------- Loss Function --------
        if cli_args.loss == "mse":
            self.loss_fn = MSELoss()
        elif cli_args.loss == "cross_entropy":
            self.loss_fn = CrossEntropyLoss()
        else:
            raise ValueError("Unsupported loss function")

        # -------- Optimizer --------
        lr = cli_args.learning_rate
        wd = cli_args.weight_decay

        if cli_args.optimizer == "sgd":
            self.optimizer = SGD(lr,wd)

        elif cli_args.optimizer == "momentum":
            self.optimizer = Momentum(lr, wd)

        elif cli_args.optimizer == "nag":
             self.optimizer = NAG(lr, wd)

        elif cli_args.optimizer == "rmsprop":
            self.optimizer = RMSProp(lr, wd)

        # elif cli_args.optimizer == "adam":
        #     self.optimizer = Adam(lr, wd)

        # elif cli_args.optimizer == "nadam":
        #     self.optimizer = Nadam(lr, wd)

        else:
            raise ValueError("Unsupported optimizer")

    def forward(self, X):
        """
        Forward propagation through all layers.

        Args:
            X: Input data

        Returns:
            Output logits
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # def backward(self, y_true, y_pred):
    def backward(self):
        grad_list = []
        grad = self.loss_fn.backward()
        grad_list.append(grad)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            grad_list.append(grad)
        return grad_list
    
    def get_weights(self):
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                weights[i] = {'W': layer.W, 'b': layer.b}
        return weights

    def set_weights(self, weights):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W') and i in weights:
                layer.W = weights[i]['W']
                layer.b = weights[i]['b']

    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.step(self.layers)

    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0.0

            for i in range(0, n_samples, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward pass
                logits = self.forward(X_batch)

                # Compute loss
                if self.cli_args.loss == "mse":
                    y_input = np.eye(10)[y_batch]   # one-hot
                else:
                    y_input = y_batch               # integer labels for cross-entropy
                loss = self.loss_fn.forward(logits, y_input)
                epoch_loss += loss

                # Backward pass
                # self.backward(y_batch, logits)
                grad_list = self.backward()

                # Update parameters
                self.update_weights()

            # avg_loss = epoch_loss / (n_samples // batch_size)              # ignores remainder samples
            avg_loss = epoch_loss / math.ceil(n_samples / batch_size)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss
            })

    # def evaluate(self, X, y):
    #     """
    #     Evaluate the network on given data.
    #     """
    #     logits = self.forward(X)
    #     y_pred = np.argmax(logits, axis=1)
    #     accuracy = np.mean(y_pred == y)
    #     return accuracy
    
    def evaluate(self, X, y):
        logits = self.forward(X)

        # Loss
        loss = self.loss_fn.forward(logits, y)

        # Predictions
        y_pred = np.argmax(logits, axis=1)

        # Accuracy
        accuracy = np.mean(y_pred == y)

        # Precision, Recall, F1 (macro-averaged)
        num_classes = np.max(y) + 1
        precision_list = []
        recall_list = []

        for c in range(num_classes):
            tp = np.sum((y_pred == c) & (y == c))
            fp = np.sum((y_pred == c) & (y != c))
            fn = np.sum((y_pred != c) & (y == c))

            precision_c = tp / (tp + fp + 1e-8)
            recall_c = tp / (tp + fn + 1e-8)

            precision_list.append(precision_c)
            recall_list.append(recall_c)

        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return accuracy, precision, recall, f1, loss



            


# class NeuralNetwork:
#     """
#     Main model class that orchestrates the neural network training and inference.
#     """
    
#     def __init__(self, cli_args):
#         """
#         Initialize the neural network.

#         Args:
#             cli_args: Command-line arguments for configuring the network
#         """
#         pass
    
#     def forward(self, X):
#         """
#         Forward propagation through all layers.
        
#         Args:
#             X: Input data
            
#         Returns:
#             Output logits
#         """
#         pass
    
#     def backward(self, y_true, y_pred):
#         """
#         Backward propagation to compute gradients.
        
#         Args:
#             y_true: True labels
#             y_pred: Predicted outputs
            
#         Returns:
#             return grad_w, grad_b
#         """
#         pass
    
#     def update_weights(self):
#         """
#         Update weights using the optimizer.
#         """
#         pass
    
#     def train(self, X_train, y_train, epochs, batch_size):
#         """
#         Train the network for specified epochs.
#         """
#         pass
    
#     def evaluate(self, X, y):
#         """
#         Evaluate the network on given data.
#         """
#         pass
