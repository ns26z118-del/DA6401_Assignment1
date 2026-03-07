import argparse
import ast
import numpy as np
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from ann.objective_functions import MSELoss


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("-d", "--dataset",
                        choices=["mnist", "fashion_mnist"],
                        default="mnist")

    parser.add_argument("-e", "--epochs", type=int, default=10)

    parser.add_argument("-b", "--batch_size", type=int, default=64)   # best config

    parser.add_argument("-l", "--loss",
                        choices=["mse", "cross_entropy"],
                        default="cross_entropy")

    parser.add_argument("-o", "--optimizer",
                        choices=["sgd", "momentum", "nag", "rmsprop"],
                        default="rmsprop")               # best config

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.005)  # best config

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0001)  # best config

    parser.add_argument("-nhl", "--num_layers", type=int, default=2)          # best config

    parser.add_argument("-sz", "--hidden_size",
                        type=str,                        # matches train.py
                        default="[128]")                 # best config

    parser.add_argument("-a", "--activation",
                        choices=["relu", "sigmoid", "tanh"],
                        default="tanh")                  # best config

    parser.add_argument("-wi", "--weight_init",
                        choices=["random", "xavier"],
                        default="xavier")

    parser.add_argument("-wp", "--wandb_project",
                        default="da6401-assignment-1")

    parser.add_argument("--model_save_path", default="models/model.npy")

    return parser.parse_args()


def load_model(model_path, model):
    data = np.load(model_path, allow_pickle=True).item()
    model.set_weights(data)
    return model


def evaluate_model(model, X_test, y_test):
    logits = model.forward(X_test)
    y_pred = np.argmax(logits, axis=1)

    accuracy = np.mean(y_pred == y_test)


    if isinstance(model.loss_fn, MSELoss):
        y_input = np.eye(10)[y_test]
    else:
        y_input = y_test
    loss = model.loss_fn.forward(logits, y_input)

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

    # Match train.py hidden_size parsing
    if isinstance(args.hidden_size, str):
        args.hidden_size = ast.literal_eval(args.hidden_size)

    _, _, X_test, y_test = load_data(args.dataset)

    model = NeuralNetwork(args)
    model = load_model(args.model_path, model)

    results = evaluate_model(model, X_test, y_test)

    print("Evaluation Results:")
    for k, v in results.items():
        if k != "logits":
            print(f"{k}: {v}")

    return results


if __name__ == "__main__":
    main()


# import argparse
 
# import numpy as np

# from utils.data_loader import load_data
# from ann.neural_network import NeuralNetwork
# from ann.objective_functions import MSELoss, CrossEntropyLoss


# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Run inference on test set")

#     parser.add_argument("--model_path", type=str, required=True)

#     parser.add_argument("-d", "--dataset",
#                         choices=["mnist", "fashion_mnist"],
#                         default="mnist")

#     parser.add_argument("-b", "--batch_size",
#                         type=int,
#                         default=32)

#     parser.add_argument("-l", "--loss",
#                         choices=["mse", "cross_entropy"],
#                         default="cross_entropy")

#     parser.add_argument("-o", "--optimizer",
#                         choices=["sgd", "momentum", "nag", "rmsprop"], # Addnl: "adam", "nadam"
#                         default="sgd")

#     parser.add_argument("-lr", "--learning_rate",
#                         type=float,
#                         default=0.001)

#     parser.add_argument("-wd", "--weight_decay",
#                         type=float,
#                         default=0.0)

#     parser.add_argument("-nhl", "--num_layers",
#                         type=int,
#                         default=1)

#     parser.add_argument("-sz", "--hidden_size",
#                         nargs="+",
#                         type=int,
#                         default="[128]")

#     parser.add_argument("-a", "--activation",
#                         choices=["relu", "sigmoid", "tanh"],
#                         default="relu")

#     parser.add_argument("-wi", "--weight_init",
#                         choices=["random", "xavier"],
#                         default="xavier")

#     return parser.parse_args()


# def load_model(model_path, model):
#     data = np.load(model_path, allow_pickle=True).item()
#     model.set_weights(data)
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
#     if isinstance(model.loss_fn, MSELoss):
#         y_input = np.eye(10)[y_test]
#     else:
#         y_input = y_test
#     loss = model.loss_fn.forward(logits, y_test)

#     # Precision, Recall, F1
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
#     args = parse_arguments()

  
#     _, _, X_test, y_test = load_data(args.dataset)

    
#     model = NeuralNetwork(args) 

 
#     model = load_model(args.model_path, model)

     
#     results = evaluate_model(model, X_test, y_test)

#     print("Evaluation Results:")
#     for k, v in results.items():
#         if k != "logits":
#             print(f"{k}: {v}")

#     return results

# if __name__ == "__main__":
#     main()
