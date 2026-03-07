import numpy as np

class NeuralLayer:
    def __init__(self, in_features, out_features, weight_init):

        self.in_features = in_features
        self.out_features = out_features

        if weight_init == "xavier":
            limit = np.sqrt(6 / (in_features + out_features))
            self.W = np.random.uniform(-limit, limit, (in_features, out_features))

        elif weight_init == "random":
            self.W = 0.01 * np.random.randn(in_features, out_features)

        else:
            raise ValueError("weight_init must be 'xavier' or 'random'")

 
        self.b = np.zeros((1, out_features))

        # Gradients 
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
