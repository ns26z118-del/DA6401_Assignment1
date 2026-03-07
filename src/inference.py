"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
 
import numpy as np

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("-d", "--dataset",
                        choices=["mnist", "fashion_mnist"],
                        default="mnist")

    parser.add_argument("-b", "--batch_size",
                        type=int,
                        default=32)

    parser.add_argument("-l", "--loss",
                        choices=["mse", "cross_entropy"],
                        default="cross_entropy")

    parser.add_argument("-o", "--optimizer",
                        choices=["sgd", "momentum", "nag", "rmsprop"], # Addnl: "adam", "nadam"
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

    return parser.parse_args()

# def load_model(model_path, model):
#     """
#     Load trained weights and assign them to model layers.
#     """

#     weights = np.load(model_path, allow_pickle=True)

#     idx = 0
#     for layer in model.layers:
#         if hasattr(layer, "W"):
#             layer.W = weights[idx]
#             layer.b = weights[idx + 1]
#             idx += 2

#     return model

def load_model(model_path, model):
    data = np.load(model_path, allow_pickle=True).item()
    model.set_weights(data)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.

    Returns Dictionary:
    logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)
    y_pred = np.argmax(logits, axis=1)

    # Accuracy
    accuracy = np.mean(y_pred == y_test)

    # Loss
    loss = model.loss_fn.forward(logits, y_test)

    # Precision, Recall, F1
    num_classes = len(np.unique(y_test))
    precision_list, recall_list, f1_list = [], [], []

    for cls in range(num_classes):
        tp = np.sum((y_pred == cls) & (y_test == cls))
        fp = np.sum((y_pred == cls) & (y_test != cls))
        fn = np.sum((y_pred != cls) & (y_test == cls))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list)),
        "f1": float(np.mean(f1_list))
    }


def main():
    args = parse_arguments()

    # Load dataset
    _, _, X_test, y_test = load_data(args.dataset)

    # Recreate model architecture
    model = NeuralNetwork(args) 

    # Load trained weights
    model = load_model(args.model_path, model)

    # Evaluate
    results = evaluate_model(model, X_test, y_test)

    print("Evaluation Results:")
    for k, v in results.items():
        if k != "logits":
            print(f"{k}: {v}")

    return results

if __name__ == "__main__":
    main()

# """
# Inference Script
# Evaluate trained models on test sets
# """

# import argparse
# import pickle
# import numpy as np

# from utils.data_loader import load_data


# def parse_arguments():
#     """
#     Parse command-line arguments for inference.
#     """
#     parser = argparse.ArgumentParser(description="Run inference on test set")

#     parser.add_argument("--model_path", type=str, required=True,
#                         help="Relative path to saved model")

#     parser.add_argument("--dataset", type=str, default="mnist",
#                         choices=["mnist", "fashion_mnist"])

#     parser.add_argument("--batch_size", type=int, default=64)

#     parser.add_argument("--hidden_layers", type=int, default=1)

#     parser.add_argument("--num_neurons", type=int, default=128)

#     parser.add_argument("--activation", type=str, default="relu",
#                         choices=["relu", "sigmoid", "tanh"])

#     return parser.parse_args()


# def load_model(model_path):
#     """
#     Load trained model from disk.
#     """
#     with open(model_path, "rb") as f:
#         model = pickle.load(f)
#     return model


# def evaluate_model(model, X_test, y_test):
#     """
#     Evaluate model on test data.

#     Returns Dictionary:
#     logits, loss, accuracy, f1, precision, recall
#     """
#     logits = model.forward(X_test)
#     y_pred = np.argmax(logits, axis=1)

#     # Accuracy
#     accuracy = np.mean(y_pred == y_test)

#     # Loss
#     loss = model.loss_fn.forward(logits, y_test)

#     # Precision, Recall, F1 (macro averaged)
#     num_classes = len(np.unique(y_test))
#     precision_list, recall_list, f1_list = [], [], []

#     for cls in range(num_classes):
#         tp = np.sum((y_pred == cls) & (y_test == cls))
#         fp = np.sum((y_pred == cls) & (y_test != cls))
#         fn = np.sum((y_pred != cls) & (y_test == cls))

#         precision = tp / (tp + fp + 1e-8)
#         recall = tp / (tp + fn + 1e-8)
#         f1 = 2 * precision * recall / (precision + recall + 1e-8)

#         precision_list.append(precision)
#         recall_list.append(recall)
#         f1_list.append(f1)

#     return {
#         "logits": logits,
#         "loss": loss,
#         "accuracy": accuracy,
#         "precision": float(np.mean(precision_list)),
#         "recall": float(np.mean(recall_list)),
#         "f1": float(np.mean(f1_list))
#     }


# def main():
#     """
#     Main inference function.
#     """
#     args = parse_arguments()

#     # Load dataset
#     _, _, X_test, y_test = load_data(args.dataset)

#     # Load model
#     model = load_model(args.model_path)

#     # Evaluate
#     results = evaluate_model(model, X_test, y_test)

#     print("Evaluation Results:")
#     for k, v in results.items():
#         if k != "logits":
#             print(f"{k}: {v}")

#     return results


# if __name__ == "__main__":
#     main()
# import argparse

# def parse_arguments():
#     """
#     Parse command-line arguments for inference.
    
#     TODO: Implement argparse with:
#     - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
#     - dataset: Dataset to evaluate on
#     - batch_size: Batch size for inference
#     - hidden_layers: List of hidden layer sizes
#     - num_neurons: Number of neurons in hidden layers
#     - activation: Activation function ('relu', 'sigmoid', 'tanh')
#     """
#     parser = argparse.ArgumentParser(description='Run inference on test set')
    
#     return parser.parse_args()


# def load_model(model_path):
#     """
#     Load trained model from disk.
#     """
#     pass


# def evaluate_model(model, X_test, y_test): 
#     """
#     Evaluate model on test data.
        
#     TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
#     """
#     pass


# def main():
#     """
#     Main inference function.

#     TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
#     """
#     args = parse_arguments()
    
#     print("Evaluation complete!")


# if __name__ == '__main__':
#     main()
