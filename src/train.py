"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""


import argparse
import os
import pickle
import wandb
import numpy as np
import json
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument("-d", "--dataset",
                        choices=["mnist", "fashion_mnist"],
                        default="mnist")

    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=10)

    parser.add_argument("-b", "--batch_size",
                        type=int,
                        default=32)

    parser.add_argument("-l", "--loss",
                        choices=["mse", "cross_entropy"],
                        default="cross_entropy")

    parser.add_argument("-o", "--optimizer",
                        choices=["sgd", "momentum", "nag", "rmsprop"],
                        default="sgd")

    parser.add_argument("-lr", "--learning_rate",
                        type=float,
                        default=0.001)

    parser.add_argument("-wd", "--weight_decay",
                        type=float,
                        default=0.0)

    parser.add_argument("-nhl", "--num_layers",
                        type=int,
                        default=1)

    parser.add_argument("-sz", "--hidden_size",
                        nargs="+",
                        type=int,
                        default=[128])

    parser.add_argument("-a", "--activation",
                        choices=["relu", "sigmoid", "tanh"],
                        default="relu")

    parser.add_argument("-wi", "--weight_init",
                        choices=["random", "xavier"],
                        default="xavier")

    parser.add_argument("-wp", "--wandb_project",
                        default="da6401-assignment-1")

    parser.add_argument("--model_save_path",
                        default="models/model.npy")

    return parser.parse_args()

def main():
    """
    Main training function.
    """
    args = parse_arguments()

    # Expand hidden layer sizes
    # args.hidden_size = [args.num_neurons] * args.hidden_layers

    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        config=vars(args)
    )

    # Load dataset
    X_train, y_train, X_test, y_test = load_data(args.dataset)

    # Initialize model
    model = NeuralNetwork(args)

    # Train model
    model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Evaluate
    # accuracy = model.evaluate(X_test, y_test)
    accuracy, loss, precision, recall, f1 = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # wandb.log({"test_accuracy": accuracy})
    wandb.log({
    "test_accuracy": accuracy,
    "test_loss": loss,
    "precision": precision,
    "recall": recall,
    "f1": f1
    })

    # Save model (relative path only)
    # os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    # with open(args.model_save_path, "wb") as f:
    #     pickle.dump(model, f)

    # Save model weights as .npy

    # best_f1 = -1
    # if f1 > best_f1:
    # best_f1 = f1

    # weights = []
    # for layer in model.layers:
    #     if hasattr(layer, "W"):
    #         weights.append(layer.W)
    #         weights.append(layer.b)

    # weights = np.array(weights, dtype=object)
    # np.save("best_model.npy", weights)

    # print("New best model saved!")


    weights = []

    for layer in model.layers:
        if hasattr(layer, "W"):
            weights.append(layer.W)
            weights.append(layer.b)

    weights = np.array(weights, dtype=object)

    np.save("best_model.npy", weights)

    print("Model saved as best_model.npy")

    with open("best_config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    print("Training complete!")


if __name__ == "__main__":
    main()


# old parser 

# def parse_arguments():
#     """
#     Parse command-line arguments.
#     """
#     parser = argparse.ArgumentParser(description="Train a neural network")

#     parser.add_argument("--dataset", type=str, default="mnist",
#                         choices=["mnist", "fashion_mnist"])

#     parser.add_argument("--epochs", type=int, default=10)

#     parser.add_argument("--batch_size", type=int, default=32)

#     parser.add_argument("--learning_rate", type=float, default=0.001)

#     parser.add_argument("--optimizer", type=str, default="sgd",
#                         choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])

#     parser.add_argument("--hidden_layers", type=int, default=1)

#     parser.add_argument("--num_neurons", type=int, default=128)

#     parser.add_argument("--activation", type=str, default="relu",
#                         choices=["relu", "sigmoid", "tanh"])

#     parser.add_argument("--loss", type=str, default="cross_entropy",
#                         choices=["cross_entropy", "mse"])

#     parser.add_argument("--weight_init", type=str, default="xavier")

#     parser.add_argument("--wandb_project", type=str, default="da6401-assignment-1")

#     parser.add_argument("--model_save_path", type=str, default="models/model.pkl")

#     return parser.parse_args()


# import argparse

# def parse_arguments():
#     """
#     Parse command-line arguments.
    
#     TODO: Implement argparse with the following arguments:
#     - dataset: 'mnist' or 'fashion_mnist'
#     - epochs: Number of training epochs
#     - batch_size: Mini-batch size
#     - learning_rate: Learning rate for optimizer
#     - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
#     - hidden_layers: List of hidden layer sizes
#     - num_neurons: Number of neurons in hidden layers
#     - activation: Activation function ('relu', 'sigmoid', 'tanh')
#     - loss: Loss function ('cross_entropy', 'mse')
#     - weight_init: Weight initialization method
#     - wandb_project: W&B project name
#     - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
#     """
#     parser = argparse.ArgumentParser(description='Train a neural network')
    
#     return parser.parse_args()


# def main():
#     """
#     Main training function.
#     """
#     args = parse_arguments()
    
#     print("Training complete!")


# if __name__ == '__main__':
#     main()
