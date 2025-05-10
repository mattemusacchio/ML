from typing import Tuple
import cupy as np

def train_val_test_split(X: np.ndarray, y: np.ndarray,
                         train_size: float = 0.7,
                         val_size: float = 0.15,
                         test_size: float = 0.15,
                         random_state: int = 42) -> Tuple:
    assert np.isclose(train_size + val_size + test_size, 1.0), "Las proporciones deben sumar 1"

    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X, y = X[indices], y[indices]
    n_train = int(train_size * len(X))
    n_val = int(val_size * len(X))

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test

def one_hot_encode(y, num_classes=None):
    if num_classes is None:
        num_classes = np.max(y) + 1
    return np.eye(num_classes)[y]

