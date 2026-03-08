"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
import os

from utils.data_loader import load_and_prep_data
from ann.neural_network import NeuralNetwork

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    parser.add_argument('--model_path', type=str, default='best_model.npy')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('--batch_size', type=int, default=128)
    
    # CRITICAL FIX: Updated to match the autograder
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, nargs='+', default=[128, 128, 128])
    
    parser.add_argument('--activation', type=str, default='tanh', choices=['relu', 'sigmoid', 'tanh'])
    
    return parser.parse_args()

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    return np.load(model_path, allow_pickle=True).item()

def evaluate_model(model, X_test, y_test): 
    return model.evaluate(X_test, y_test)

def main():
    args = parse_arguments()
    
    print(f"Loading {args.dataset} test data...")
    _, _, _, _, X_test, y_test = load_and_prep_data(args.dataset)
    
    print("Reconstructing Neural Network architecture...")
    model = NeuralNetwork(args)
    
    print(f"Loading weights from {args.model_path}...")
    weights = load_model(args.model_path)
    model.set_weights(weights)
    
    print("Evaluating model on test data...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\n" + "="*35)
    print("FINAL INFERENCE RESULTS")
    print("="*35)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Loss:      {metrics['loss']:.4f}")
    print("\n")
    
    return metrics

if __name__ == '__main__':
    main()
