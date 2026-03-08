import numpy as np

class Activation:
    def __init__(self, activation):

        self.activation = activation

    def forward(self, Z):
        self.Z = Z  # cache for backward

        if self.activation == "relu":
            return np.maximum(0, Z)

        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-Z))

        elif self.activation == "tanh":
            return np.tanh(Z)

        else:
            raise ValueError("Unsupported activation")

    def backward(self, dA):
   
        if self.activation == "relu":
            dZ = dA * (self.Z > 0)

        elif self.activation == "sigmoid":
            sig = 1 / (1 + np.exp(-self.Z))
            dZ = dA * sig * (1 - sig)

        elif self.activation == "tanh":
            t = np.tanh(self.Z)
            dZ = dA * (1 - t**2)

        return dZ
    

