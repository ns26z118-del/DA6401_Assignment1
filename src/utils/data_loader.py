"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split

def load_dataset(name):
    test_size = 0.2
    val_size = 0.2
    random_state = 42

    if name == "mnist":
        (X, y), _ = mnist.load_data()
    elif name == "fashion_mnist":
        (X, y), _ = fashion_mnist.load_data()

    X = X.reshape(X.shape[0], 784) / 255.0

    #train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    #train validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train
    )

    return X_train, y_train, X_val, y_val, X_test, y_test