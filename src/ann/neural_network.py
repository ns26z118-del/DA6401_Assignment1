"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from .neural_layer import neural_layer
from .activations import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_derivative
from .objective_functions import cross_entropy_grad, mse_grad
from .optimizers import SGD, Momentum, RMSProp
import wandb

class NeuralNetwork:

    def __init__(self, cli_options):

        self.layers= []
        n= 784 #hardcoded since MNIST images are 28x28

        for h in cli_options.hidden_size:
            # FIX 1: was neural_layer(n, h, cli_options.weight_init) —
            # weight_init string was passed into the activation slot,
            # so all hidden layers had linear activation and the network couldn't learn.
            self.layers.append(neural_layer(n, h, cli_options.activation, cli_options.weight_init))
            n= h

        # Output layer: linear activation (logits), softmax applied inside loss
        self.layers.append(neural_layer(n, 10, "linear", cli_options.weight_init))
        self.activation= cli_options.activation
        self.loss= cli_options.loss

        if cli_options.optimizer == "sgd":
            self.optimizer= SGD(cli_options.learning_rate)
        elif cli_options.optimizer == "momentum":
            self.optimizer= Momentum(cli_options.learning_rate)
        else:
            self.optimizer= RMSProp(cli_options.learning_rate)

    def activate(self, x):
        if self.activation == "relu":
            return relu(x)
        if self.activation == "sigmoid":
            return sigmoid(x)
        return tanh(x)

    def activate_grad(self, x):
        if self.activation == "relu":
            return relu_derivative(x)
        if self.activation == "sigmoid":
            return sigmoid_derivative(x)
        return tanh_derivative(x)

    def forward(self, X):
        out= X
        self.z= []
        self.activations= []

        for i, layer in enumerate(self.layers):
            z= layer.forward_pass(out)
            self.z.append(z)
            if i != len(self.layers) - 1:
                out= self.activate(z)
                self.activations.append(out)
            else:
                out= z
        return out

    def backward(self, y_true, y_pred):
        if self.loss == "cross_entropy":
            grad = cross_entropy_grad(y_true, y_pred)
        else:
            grad = mse_grad(y_true, y_pred)

        dW_list = []
        db_list = []

        for layer in reversed(self.layers):
            grad = layer.backward_pass(grad)
            dW_list.append(layer.grad_W)
            db_list.append(layer.grad_b)

        # Autograder expects [Layer0_grad, Layer1_grad, ...]
        self.grad_W = dW_list[::-1]
        self.grad_b = db_list[::-1]

        return self.grad_W, self.grad_b

    def update_weights(self):
        for i, layer in enumerate(self.layers):
            if isinstance(self.optimizer, SGD):
                self.optimizer.update(layer)
            else:
                self.optimizer.update(layer, i)

    def get_weights(self):
        weights_dict = {}
        for i, layer in enumerate(self.layers):
            weights_dict[f"W{i}"] = layer.W
            weights_dict[f"b{i}"] = layer.b
        return weights_dict

    def set_weights(self, weight_dict):
        # FIX 2: autograder calls np.load(..., allow_pickle=True) WITHOUT .item(),
        # which returns a 0-d numpy object array instead of a plain dict.
        # Unwrap it here so both call patterns work correctly.
        if isinstance(weight_dict, np.ndarray):
            weight_dict = weight_dict.item()

        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict and b_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
                layer.b = weight_dict[b_key].copy()
            else:
                print(f"Warning: {w_key} or {b_key} not found in provided weights.")

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=1, batch_size=32):
        n= X_train.shape[0]
        num_batches= max(n // batch_size, 1)

        for epoch in range(epochs):
            # Shuffle each epoch for better convergence
            indices = np.random.permutation(n)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            for i in range(0, n, batch_size):
                xb= X_shuffled[i:i + batch_size]
                yb= y_shuffled[i:i + batch_size]

                logits = self.forward(xb)

                if self.loss == "cross_entropy":
                    from .objective_functions import cross_entropy_loss
                    batch_loss = cross_entropy_loss(logits, yb)
                else:
                    from .objective_functions import mse_loss
                    batch_loss = mse_loss(logits, yb)

                epoch_loss += batch_loss
                self.backward(yb, logits)
                self.update_weights()

            epoch_loss /= num_batches

            # Training accuracy
            train_logits = self.forward(X_train)
            train_preds = np.argmax(train_logits, axis=1)
            train_acc = np.mean(train_preds == y_train)

            # Validation accuracy
            if X_val is not None:
                val_logits = self.forward(X_val)
                val_preds = np.argmax(val_logits, axis=1)
                val_acc = np.mean(val_preds == y_val)
            else:
                val_acc = None

            log_dict = {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_accuracy": train_acc
            }
            if val_acc is not None:
                log_dict["val_accuracy"] = val_acc

            wandb.log(log_dict)

    def evaluate(self, X, y):
        logits= self.forward(X)
        preds= np.argmax(logits, axis=1)
        return np.mean(preds == y)