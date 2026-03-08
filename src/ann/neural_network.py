import math
import numpy as np
import wandb
from ann.neural_layer import NeuralLayer
from ann.activations import Activation
from ann.objective_functions import MSELoss, CrossEntropyLoss
from ann.optimizers import SGD, Momentum, NAG, RMSProp

class NeuralNetwork:

    def __init__(self, cli_args):
        self.cli_args = cli_args
        self.layers = []

        input_dim = 784
        num_classes = 10
        prev_dim = input_dim

        for hidden_dim in cli_args.hidden_size:
            self.layers.append(NeuralLayer(prev_dim, hidden_dim, cli_args.weight_init))
            self.layers.append(Activation(cli_args.activation))
            prev_dim = hidden_dim

        self.layers.append(NeuralLayer(prev_dim, num_classes, cli_args.weight_init))

        if cli_args.loss == "mse":
            self.loss_fn = MSELoss()
        elif cli_args.loss == "cross_entropy":
            self.loss_fn = CrossEntropyLoss()
        else:
            raise ValueError("Unsupported loss function")

        lr = cli_args.learning_rate
        wd = cli_args.weight_decay

        if cli_args.optimizer == "sgd":
            self.optimizer = SGD(lr, wd)
        elif cli_args.optimizer == "momentum":
            self.optimizer = Momentum(lr, wd)
        elif cli_args.optimizer == "nag":
            self.optimizer = NAG(lr, wd)
        elif cli_args.optimizer == "rmsprop":
            self.optimizer = RMSProp(lr, wd)
        else:
            raise ValueError("Unsupported optimizer")

    def forward(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, logits=None, y_true=None):
        """
        Backward pass. Supports both:
          model.backward()               -- normal training (loss already computed)
          model.backward(logits, y_true) -- autograder direct call
        """
        if logits is not None and y_true is not None:
            if logits.ndim == 1:
                logits = logits.reshape(1, -1)

            if self.cli_args.loss == "mse":
                # MSE expects one-hot y_true
                if y_true.ndim == 1 and y_true.dtype in [np.int32, np.int64] and y_true.max() < 10:
                    y_input = np.eye(10)[y_true.astype(int)]
                else:
                    y_input = y_true
            else:
                # CrossEntropy expects integer class indices
                if y_true.ndim == 2:
                    y_input = np.argmax(y_true, axis=1)
                else:
                    y_input = np.array(y_true).flatten().astype(int)

            self.loss_fn.forward(logits, y_input)

        grad = self.loss_fn.backward()
        grad_list = [grad]
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            grad_list.append(grad)
        return grad_list

    def get_weights(self):
        """
        Returns weights keyed by weight-layer index (0-based, skipping Activation layers).
        """
        weights = {}
        weight_idx = 0
        for layer in self.layers:
            if hasattr(layer, 'W'):
                weights[weight_idx] = {'W': layer.W, 'b': layer.b}
                weight_idx += 1
        return weights

    def set_weights(self, weights):
        """
        Loads weights by weight-layer index (matching get_weights format).
        Handles both integer keys and string keys (e.g. '0', '1').
        """
        weight_idx = 0
        for layer in self.layers:
            if hasattr(layer, 'W'):
                # Accept both int key and string key
                key = weight_idx if weight_idx in weights else str(weight_idx)
                if key in weights:
                    layer.W = weights[key]['W']
                    layer.b = weights[key]['b']
                weight_idx += 1

    def update_weights(self):
        self.optimizer.step(self.layers)

    def train(self, X_train, y_train, epochs, batch_size):
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0.0

            for i in range(0, n_samples, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                logits = self.forward(X_batch)

                if self.cli_args.loss == "mse":
                    y_input = np.eye(10)[y_batch.astype(int)]
                else:
                    y_input = y_batch
                loss = self.loss_fn.forward(logits, y_input)
                epoch_loss += loss

                self.backward()
                self.update_weights()

            avg_loss = epoch_loss / math.ceil(n_samples / batch_size)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

    def evaluate(self, X, y):
        logits = self.forward(X)

        # Compute loss correctly for each loss type
        if self.cli_args.loss == "mse":
            y_input = np.eye(10)[y.astype(int)]
        else:
            y_input = y
        loss = self.loss_fn.forward(logits, y_input)

        y_pred = np.argmax(logits, axis=1)
        accuracy = np.mean(y_pred == y)

        num_classes = 10
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