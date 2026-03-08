"""
Inference Script
Evaluate trained models on test sets
"""
import argparse
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import os
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt

def parse_arguments():
    parser= argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0001)
    parser.add_argument("-l", "--loss", type=str, choices=["mse", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop"], default="rmsprop")
    parser.add_argument("-nhl", "--num_layers", type=int, default=2)
    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, default=[96, 96])
    parser.add_argument("-a", "--activation", type=str, choices=["sigmoid", "tanh", "relu"], default="sigmoid")
    parser.add_argument("-wi", "--weight_init", type=str, choices=["random", "xavier"], default="xavier")
    parser.add_argument("-wp", "--wandb_project", type=str, default="da6401")
    parser.add_argument("-m", "--model_path", type=str)

    return parser.parse_args()

def get_model_path(args):
    if args.model_path:
        return args.model_path

    base_name = f"{args.dataset}_epochs{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}_opt_{args.optimizer}_hl{'-'.join(map(str, args.hidden_size))}_act_{args.activation}_winit_{args.weight_init}"
    folder = os.path.join("..", "models")
    save_path = os.path.join(folder, f"{base_name}.npy")

    if os.path.exists(save_path):
        return save_path

    i = 1
    while True:
        numbered_path = save_path.replace(".npy", f"_{i}.npy")
        if os.path.exists(numbered_path):
            i += 1
        else:
            break
    if i > 1:
        return save_path.replace(".npy", f"_{i-1}.npy")

    print(f"No model found for given hyperparameters in {folder}")

def load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data

def evaluate_model(model, X_test, y_test):
    logits = model.forward(X_test)
    preds = np.argmax(logits, axis=1)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall": recall_score(y_test, preds, average="macro", zero_division=0),
        "f1": f1_score(y_test, preds, average="macro", zero_division=0)
    }

    cm = confusion_matrix(y_test, preds, labels=np.arange(10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

 
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    plt.close()

    return metrics

def main():
    args = parse_arguments()

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args.dataset)
    model_path = get_model_path(args)

    model = NeuralNetwork(args)
    weights = load_model(model_path)
    model.set_weights(weights)

    results = evaluate_model(model, X_test, y_test)
    print(f"Model evaluated= {model_path}")
    print(results)

if __name__ == "__main__":
    main()