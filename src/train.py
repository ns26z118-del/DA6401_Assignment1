import argparse
import os
import pickle
import wandb
import numpy as np
import json
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
from sklearn.model_selection import train_test_split

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




def log_sample_images(X, y):
    import wandb
    from collections import defaultdict

    table = wandb.Table(columns=["image", "label"])
    class_count = defaultdict(int)

    for img, label in zip(X, y):
        if class_count[label] < 5:
            image = img.reshape(28, 28)  # MNIST / FashionMNIST
            table.add_data(wandb.Image(image), int(label))
            class_count[label] += 1

        if all(v >= 5 for v in class_count.values()):
            break

    wandb.log({"sample_images_per_class": table})

def main():

    args = parse_arguments()

    wandb.init(
        project=args.wandb_project,
        config=vars(args)
    )

    # Load dataset
    X_train_full, y_train_full, _, _ = load_data(args.dataset)

    # Split train and val datasets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)


    # Log dataset samples to W&B
    log_sample_images(X_train_full, y_train_full)

    # Initialize model
    model = NeuralNetwork(args)

    # Train model
    model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    accuracy, precision, recall, f1, loss = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation f1-score: {f1:.4f}")


    wandb.log({
    "test_accuracy": accuracy,
    "test_loss": loss,
    "precision": precision,
    "recall": recall,
    "f1": f1
    })

    best = False
    os.makedirs(f"models/{args.dataset}", exist_ok=True)
    try:
        with open(f"models/{args.dataset}/best_config.json", "r") as f:
            json_data = json.load(f)
        if json_data["metrics"]["f1"] < f1:
            best = True
    except:
        best = True

    if best:
        best_weights = model.get_weights()
        np.save(f"models/{args.dataset}/best_model.npy", best_weights)

        data = {
            "config": vars(args),
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        }

        with open(f"models/{args.dataset}/best_config.json", "w") as f:
            json.dump(data, f, indent=4)
        
        print("Model saved as best_model.npy")

    print("Training complete!")


if __name__ == "__main__":
    main()


