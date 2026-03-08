import numpy as np
import os
import urllib.request
import tempfile
from sklearn.model_selection import train_test_split

def load_and_prep_data(dataset_name='mnist'):
    # Ultra-fast direct download to bypass timeouts entirely
    if dataset_name.lower() == 'mnist':
        filepath = os.path.join(tempfile.gettempdir(), "mnist.npz")
        if not os.path.exists(filepath):
            url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception:
                pass
                
        if os.path.exists(filepath):
            with np.load(filepath, allow_pickle=True) as f:
                X, y = f['x_train'], f['y_train']
                X_test, y_test = f['x_test'], f['y_test']
            X = X.reshape(X.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        else:
            from keras.datasets import mnist
            (X, y), (X_test, y_test) = mnist.load_data()
            X = X.reshape(X.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
    else:
        try:
            from keras.datasets import fashion_mnist
            (X, y), (X_test, y_test) = fashion_mnist.load_data()
            X = X.reshape(X.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        except ImportError:
            from sklearn.datasets import fetch_openml
            X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
            X, X_test, y, y_test = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)
            
    X = X / 255.0
    X_test = X_test / 255.0
    y = y.astype(int)
    y_test = y_test.astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    def one_hot(labels, num_classes=10):
        encoded = np.zeros((labels.size, num_classes))
        encoded[np.arange(labels.size), labels] = 1.0
        return encoded

    y_train_oh = one_hot(y_train)

    return X_train, y_train_oh, X_val, y_val, X_test, y_test
