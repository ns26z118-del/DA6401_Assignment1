import numpy as np

class NeuralLayer:
    def __init__(self, in_features, out_features, weight_init):
        """
        Fully connected neural layer
        Z = XW + b

        Parameters
        ----------
        in_features : int
            Number of input neurons
        out_features : int
            Number of output neurons
        weight_init : str
            "xavier" or "random"
        """

        self.in_features = in_features
        self.out_features = out_features

        # Weight initialization
        if weight_init == "xavier":
            limit = np.sqrt(6 / (in_features + out_features))
            self.W = np.random.uniform(-limit, limit, (in_features, out_features))

        elif weight_init == "random":
            self.W = 0.01 * np.random.randn(in_features, out_features)

        else:
            raise ValueError("weight_init must be 'xavier' or 'random'")

        # Bias initialization
        self.b = np.zeros((1, out_features))

        # Gradients (will be filled during backward pass)
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        """
        Forward pass

        Parameters
        ----------
        X : ndarray
            Shape (batch_size, in_features)

        Returns
        -------
        Z : ndarray
            Shape (batch_size, out_features)
        """

        self.X = X  # Cache input for backward pass
        Z = X @ self.W + self.b
        return Z

    def backward(self, dZ):
        """
        Backward pass

        Parameters
        ----------
        dZ : ndarray
            Gradient of loss w.r.t layer output
            Shape (batch_size, out_features)

        Returns
        -------
        dX : ndarray
            Gradient w.r.t input (for previous layer)
        """

        batch_size = self.X.shape[0]

        # Compute gradients
        self.grad_W = (self.X.T @ dZ) / batch_size
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / batch_size

        # Gradient for previous layer
        dX = dZ @ self.W.T

        return dX

# import numpy as np

# class NeuralLayer:
#     def __init__(self, in_features, out_features, weight_init="xavier"):
#         """
#         Fully connected neural layer: Z = XW + b
#         """
#         self.in_features = in_features
#         self.out_features = out_features

#         # Weight initialization
#         if weight_init == "xavier":
#             limit = np.sqrt(6 / (in_features + out_features))
#             self.W = np.random.uniform(-limit, limit,
#                                        (in_features, out_features))
#         else:  # random initialization
#             self.W = 0.01 * np.random.randn(in_features, out_features)

#         self.b = np.zeros(out_features)

#         # Gradients (initialized later)
#         self.grad_W = None
#         self.grad_b = None

#     def forward(self, X):
#         """
#         Forward pass
#         X: (batch_size, in_features)
#         """
#         self.X = X  # cache input for backward pass
#         return X @ self.W + self.b

#     def backward(self, dZ):
#         """
#         Backward pass
#         dZ: gradient of loss w.r.t. output Z
#             shape (batch_size, out_features)
#         """
#         # Gradient w.r.t weights
#         self.grad_W = self.X.T @ dZ

#         # Gradient w.r.t bias
#         self.grad_b = np.sum(dZ, axis=0)

#         # Gradient w.r.t input (to pass backward)
#         dX = dZ @ self.W.T
#         return dX
    



# X = np.random.randn(4, 5)
# layer = NeuralLayer(5, 3)

# Z = layer.forward(X)
# dZ = np.random.randn(4, 3)

# dX = layer.backward(dZ)

# print(layer.grad_W.shape)  # (5, 3)
# print(layer.grad_b.shape)  # (3,)
# print(dX.shape)            # (4, 5)