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
        n= 784

        for h in cli_options.hidden_size:
            self.layers.append(neural_layer(n, h, cli_options.activation, cli_options.weight_init))
            n= h

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

        # Save architecture config alongside weights so set_weights can
        # rebuild layers to the correct shape regardless of how the model
        # was constructed by the caller.
        weights_dict["__activation__"] = self.activation
        weights_dict["__num_layers__"] = len(self.layers)
        return weights_dict

    def set_weights(self, weight_dict):
        # Unwrap numpy 0-d object array (np.load without .item())
        if isinstance(weight_dict, np.ndarray):
            weight_dict = weight_dict.item()

        # Detect how many layers are in the saved weights
        num_saved_layers = sum(1 for k in weight_dict if k.startswith("W") and k[1:].isdigit())

        # If the saved model has a different number of layers than the current
        # architecture, rebuild layers to match the saved weights exactly.
        if num_saved_layers != len(self.layers):
            saved_activation = weight_dict.get("__activation__", self.activation)
            self.activation = saved_activation
            self.layers = []
            for i in range(num_saved_layers):
                W = weight_dict[f"W{i}"]
                b = weight_dict[f"b{i}"]
                n_input, n_output = W.shape
                # Last layer is always linear
                act = saved_activation if i < num_saved_layers - 1 else "linear"
                layer = neural_layer(n_input, n_output, act)
                layer.W = W.copy()
                layer.b = b.copy()
                self.layers.append(layer)
            return

        # Architecture matches — just copy weights in
        saved_activation = weight_dict.get("__activation__", self.activation)
        self.activation = saved_activation

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

            train_logits = self.forward(X_train)
            train_preds = np.argmax(train_logits, axis=1)
            train_acc = np.mean(train_preds == y_train)

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