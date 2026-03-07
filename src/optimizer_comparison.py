import argparse
import ast
import wandb
import numpy as np
from sklearn.model_selection import train_test_split
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data

OPTIMIZERS = ["sgd", "momentum", "nag", "rmsprop"]


BASE_CONFIG = {
    "dataset": "mnist",
    "epochs": 10,
    "batch_size": 32,
    "loss": "cross_entropy",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "num_layers": 3,
    "hidden_size": [128, 128, 128],   
    "activation": "relu",
    "weight_init": "xavier",
    "wandb_project": "da6401-assignment-1",
    "model_save_path": "models/model.npy"
}

def main():
    X_full, y_full, _, _ = load_data(BASE_CONFIG["dataset"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    for optimizer in OPTIMIZERS:
        config = BASE_CONFIG.copy()
        config["optimizer"] = optimizer

        run = wandb.init(
            project=config["wandb_project"],
            name=f"optimizer_{optimizer}",
            group="optimizer_showdown",   
            config=config,
            reinit=True
        )

        args = argparse.Namespace(**config)
        model = NeuralNetwork(args)
        model.train(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"])

        val_accuracy, precision, recall, f1, val_loss = model.evaluate(X_val, y_val)
        wandb.log({
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "f1": f1
        })

        print(f"[{optimizer}] val_accuracy={val_accuracy:.4f}, val_loss={val_loss:.4f}")
        run.finish()

if __name__ == "__main__":
    main()