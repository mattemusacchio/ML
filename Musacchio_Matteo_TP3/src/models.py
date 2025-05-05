import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, seed=42):
        np.random.seed(seed)
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        A = X
        activations = [X]
        zs = []

        for i in range(self.num_layers - 1):
            Z = A @ self.weights[i] + self.biases[i]
            A = relu(Z)
            zs.append(Z)
            activations.append(A)

        # Output layer
        Z = A @ self.weights[-1] + self.biases[-1]
        A = softmax(Z)
        zs.append(Z)
        activations.append(A)

        return activations, zs
    
    def backward(self, activations, zs, y_true):
        grads_w = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        m = y_true.shape[0]
        delta = activations[-1] - y_true  # Softmax + cross-entropy simplifica este paso

        for l in reversed(range(self.num_layers)):
            grads_w[l] = (activations[l].T @ delta) / m
            grads_b[l] = np.sum(delta, axis=0, keepdims=True) / m

            if l > 0:
                delta = (delta @ self.weights[l].T) * relu_derivative(zs[l - 1])

        return grads_w, grads_b
    
    def update(self, grads_w, grads_b):
        for l in range(self.num_layers):
            self.weights[l] -= self.learning_rate * grads_w[l]
            self.biases[l] -= self.learning_rate * grads_b[l]
        
    def train(self, X_train, y_train, X_val, y_val, epochs=100, verbose=True):
        history = {
            "train_loss": [],
            "val_loss": []
        }

        for epoch in range(epochs):
            # Forward pass
            activations, zs = self.forward(X_train)
            y_pred_train = activations[-1]
            loss_train = cross_entropy(y_train, y_pred_train)

            # Backward pass + update
            grads_w, grads_b = self.backward(activations, zs, y_train)
            self.update(grads_w, grads_b)

            # Validation
            val_pred = self.forward(X_val)[0][-1]
            loss_val = cross_entropy(y_val, val_pred)

            # Log
            history["train_loss"].append(loss_train)
            history["val_loss"].append(loss_val)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {loss_train:.4f}, Val Loss = {loss_val:.4f}")

        return history

import matplotlib.pyplot as plt

def plot_loss(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Evolución de la función de costo (modelo M0)")
    plt.legend()
    plt.grid(True)
    plt.show()


def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))