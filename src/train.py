import argparse
import numpy as np
import wandb
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset
import os
import json

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
    parser.add_argument("-m", "--model_save_path", type=str)

    return parser.parse_args()

def model_path(args):
    if args.model_save_path:
        return args.model_save_path 
    base= f"{args.dataset}_epochs{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}_opt_{args.optimizer}_hl{'-'.join(map(str, args.hidden_size))}_act_{args.activation}_winit_{args.weight_init}"
    loc= os.path.join("..", "models")
    os.makedirs(loc, exist_ok=True)
    result= os.path.join(loc, f"{base}.npy")

    i= 1
    orig_path= result
    while os.path.exists(result):
        result= orig_path.replace(".npy", f"_{i}.npy")
        i += 1

    return result

def main():
    args= parse_arguments()
    config_dict = vars(args)

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args.dataset) 
    wandb.init(project=args.wandb_project, config=vars(args))

    model= NeuralNetwork(args)

    # BUG FIX: was not passing X_val/y_val — validation metrics were never logged
    model.train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=args.epochs, batch_size=args.batch_size)

    acc= model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")
    wandb.log({"test_accuracy": acc})

    path= model_path(args)
    weights= model.get_weights()
    np.save(path, weights, allow_pickle=True)
    print(f"Model saved at {path}")

    config_save_path = path.replace(".npy", ".json")
    with open(config_save_path, "w") as f:
        json.dump(config_dict, f, indent=4)


if __name__ == "__main__":
    main()