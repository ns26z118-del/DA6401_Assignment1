

import os
os.environ["WANDB_MODE"] = "offline" 

import argparse
import numpy as np
import wandb

from utils.data_loader import load_and_prep_data
from ann.neural_network import NeuralNetwork

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='rmsprop', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
    
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_layers', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument('--num_neurons', type=int, default=128)
    
    parser.add_argument('--activation', type=str, default='tanh', choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'mse'])
    parser.add_argument('--weight_init', type=str, default='xavier')
    parser.add_argument('--wandb_project', type=str, default='DA6401_Assignment_1_ee21d063')
    
    parser.add_argument('--model_save_path', type=str, default='saved_model.npy')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    wandb.init(project=args.wandb_project, config=vars(args), name="Final_CLI_Train")
    
    X_train, y_train_oh, X_val, y_val, X_test, y_test = load_and_prep_data(args.dataset)
    
    model = NeuralNetwork(args)
    
    model.train(X_train, y_train_oh, epochs=args.epochs, batch_size=args.batch_size)
    
    val_metrics = model.evaluate(X_val, y_val)
    wandb.log({"final_val_accuracy": val_metrics['accuracy'], "final_val_loss": val_metrics['loss']})
    
    save_dir = os.path.dirname(args.model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    weights = model.get_weights()
    np.save(args.model_save_path, weights)
    
    wandb.finish()

if __name__ == '__main__':
    main()
