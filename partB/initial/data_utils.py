import numpy as np
from sklearn.datasets import load_digits


def get_digits():

    digits = load_digits()
    X = digits.images
    y = digits.target

    X = X / 16.0  # normalize
    X = (X - X.mean()) / X.std()

    X = np.expand_dims(X, axis=3)  # adds dimension for channel

    test_size = int(0.3 * X.shape[0])

    X_train = X[test_size:]
    y_train = y[test_size:]

    X_test = X[:test_size]
    y_test = y[:test_size]

    X_train = np.transpose(X_train, [1, 2, 3, 0])
    X_test = np.transpose(X_test, [1, 2, 3, 0])

    return X_train, y_train, X_test, y_test
