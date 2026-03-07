import numpy as np
from ann.neural_network import NeuralNetwork

# Dummy CLI args object
class Args:
    hidden_size = [64]
    activation = "relu"
    weight_init = "xavier"
    loss = "cross_entropy"
    optimizer = "sgd"
    learning_rate = 0.01
    weight_decay = 0.0

cli_args = Args()

X = np.random.randn(16, 784)
y = np.random.randint(0, 10, size=16)

model = NeuralNetwork(cli_args)
logits = model.forward(X)

model.loss_fn.forward(logits, y)
model.backward()

model.update_weights()

print("SUCCESS")