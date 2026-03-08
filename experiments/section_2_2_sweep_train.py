
import sys
sys.path.insert(0, "src")

import numpy as np
import wandb
from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork

class Args:
    def __init__(self, config):
        self.hidden_size   = config.hidden_size
        self.activation    = config.activation
        self.weight_init   = config.weight_init
        self.optimizer     = config.optimizer
        self.learning_rate = config.learning_rate
        self.loss          = config.loss
        self.dataset       = config.dataset

def main():
    run = wandb.init(project="DA6401_ns26z118")
    config = wandb.config

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(config.dataset)

    args = Args(config)
    model = NeuralNetwork(args)
    model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=config.epochs,
        batch_size=config.batch_size
    )

    test_acc = model.evaluate(X_test, y_test)
    wandb.log({"test_accuracy": test_acc})
    wandb.finish()

if __name__ == "__main__":
    main()
