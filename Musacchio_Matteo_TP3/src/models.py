import cupy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, use_adam=False, beta1=0.9, beta2=0.999, epsilon=1e-8,l2_lambda=0.0,dropout_rate=0.0, seed=42):
        np.random.seed(seed)
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []
        self.use_adam = use_adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # contador de pasos
        self.l2_lambda = l2_lambda  # Regularización L2
        self.dropout_rate = dropout_rate


        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        if use_adam:
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]


    def forward(self, X):
        A = X
        activations = [X]
        zs = []
        dropout_masks = []

        for i in range(self.num_layers - 1):
            Z = A @ self.weights[i] + self.biases[i]
            A = relu(Z)
            zs.append(Z)

            if self.dropout_rate > 0:
                mask = (np.random.rand(*A.shape) > self.dropout_rate).astype(float)
                A *= mask
                A /= (1.0 - self.dropout_rate)
                dropout_masks.append(mask)

            activations.append(A)

        # Capa de salida (sin dropout)
        Z = A @ self.weights[-1] + self.biases[-1]
        A = softmax(Z)
        zs.append(Z)
        activations.append(A)

        if self.dropout_rate > 0:
            return activations, zs, dropout_masks
        return activations, zs
    
    def backward(self, activations, zs, y_true, dropout_masks=None):
        grads_w = [None] * self.num_layers
        grads_b = [None] * self.num_layers
        m = y_true.shape[0]
        delta = activations[-1] - y_true

        for l in reversed(range(self.num_layers)):
            grads_w[l] = (activations[l].T @ delta) / m + self.l2_lambda * self.weights[l] / m
            grads_b[l] = np.sum(delta, axis=0, keepdims=True) / m

            if l > 0:
                delta = (delta @ self.weights[l].T) * relu_derivative(zs[l - 1])
                if dropout_masks and dropout_masks[l - 1] is not None:
                    delta *= dropout_masks[l - 1]
                    delta /= (1.0 - self.dropout_rate)

        return grads_w, grads_b


    
    def update(self, grads_w, grads_b):
        if not self.use_adam:
            for l in range(self.num_layers):
                self.weights[l] -= self.learning_rate * grads_w[l]
                self.biases[l] -= self.learning_rate * grads_b[l]
        else:
            self.t += 1  # paso actual
            for l in range(self.num_layers):
                # Actualizar momentos
                self.m_w[l] = self.beta1 * self.m_w[l] + (1 - self.beta1) * grads_w[l]
                self.v_w[l] = self.beta2 * self.v_w[l] + (1 - self.beta2) * (grads_w[l] ** 2)

                self.m_b[l] = self.beta1 * self.m_b[l] + (1 - self.beta1) * grads_b[l]
                self.v_b[l] = self.beta2 * self.v_b[l] + (1 - self.beta2) * (grads_b[l] ** 2)

                # Bias correction
                m_w_hat = self.m_w[l] / (1 - self.beta1 ** self.t)
                v_w_hat = self.v_w[l] / (1 - self.beta2 ** self.t)

                m_b_hat = self.m_b[l] / (1 - self.beta1 ** self.t)
                v_b_hat = self.v_b[l] / (1 - self.beta2 ** self.t)

                # Actualizar parámetros
                self.weights[l] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                self.biases[l] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        
    # def train(self, X_train, y_train, X_val, y_val, epochs=100, verbose=True):
    #     history = {
    #         "train_loss": [],
    #         "val_loss": []
    #     }

    #     for epoch in range(epochs):
    #         # Forward pass
    #         activations, zs = self.forward(X_train)
    #         y_pred_train = activations[-1]
    #         loss_train = cross_entropy(y_train, y_pred_train)

    #         # Backward pass + update
    #         grads_w, grads_b = self.backward(activations, zs, y_train)
    #         self.update(grads_w, grads_b)

    #         # Validation
    #         val_pred = self.forward(X_val)[0][-1]
    #         loss_val = cross_entropy(y_val, val_pred)

    #         # Log
    #         history["train_loss"].append(loss_train)
    #         history["val_loss"].append(loss_val)

    #         if verbose and epoch % 10 == 0:
    #             print(f"Epoch {epoch}: Train Loss = {loss_train:.4f}, Val Loss = {loss_val:.4f}")

    #     return history

    def compute_loss_with_l2(self, y_true, y_pred):
        loss = cross_entropy(y_true, y_pred)
        l2_term = sum(np.sum(W**2) for W in self.weights)
        loss += (self.l2_lambda / (2 * y_true.shape[0])) * l2_term
        return loss

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        verbose=True,
        batch_size=None,  # None = full batch (default)
        use_scheduler=False,
        scheduler_fn=None,
        early_stopping=False,
        patience=10,        # Cuántas épocas esperar antes de frenar
        min_delta=1e-4      # Cuánto tiene que mejorar la pérdida para resetear el contador
    ):
        history = {
            "train_loss": [],
            "val_loss": []
        }

        best_val_loss = float('inf')
        wait = 0  # contador para early stopping

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # ⬇️ Learning rate schedule (si se activa)
            if use_scheduler and scheduler_fn is not None:
                self.learning_rate = scheduler_fn(epoch)

            # ⬇️ Mini-batch or full-batch training
            if batch_size is None:
                # Full batch
                if self.dropout_rate > 0:
                    activations, zs, dropout_masks = self.forward(X_train)
                else:
                    activations, zs = self.forward(X_train)
                y_pred_train = activations[-1]

                if self.l2_lambda > 0:
                    loss_train = self.compute_loss_with_l2(y_train, y_pred_train)
                else:
                    loss_train = cross_entropy(y_train, y_pred_train)

                if self.dropout_rate > 0:
                    grads_w, grads_b = self.backward(activations, zs, y_train, dropout_masks=dropout_masks)
                else:
                    grads_w, grads_b = self.backward(activations, zs, y_train)
                self.update(grads_w, grads_b)
            else:
                # Mini-batch
                indices = np.random.permutation(n_samples)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
                loss_train = 0

                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    if self.dropout_rate > 0:
                        activations, zs, dropout_masks = self.forward(X_batch)
                    else:
                        activations, zs = self.forward(X_batch)
                    y_pred_batch = activations[-1]
                    loss_train += cross_entropy(y_batch, y_pred_batch)

                    if self.dropout_rate > 0:
                        grads_w, grads_b = self.backward(activations, zs, y_batch, dropout_masks=dropout_masks)
                    else:
                        grads_w, grads_b = self.backward(activations, zs, y_batch)
                    self.update(grads_w, grads_b)

                loss_train /= (n_samples // batch_size)

            # ⬇️ Validación
            y_pred_val = self.forward(X_val)[0][-1]
            loss_val = cross_entropy(y_val, y_pred_val)

            history["train_loss"].append(loss_train)
            history["val_loss"].append(loss_val)

            # ⬇️ Early stopping
            if early_stopping:
                if loss_val < best_val_loss - min_delta:
                    best_val_loss = loss_val
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break

            # ⬇️ Logging
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}: Train Loss = {loss_train:.4f}, Val Loss = {loss_val:.4f}, LR = {self.learning_rate:.6f}")

        return history



import matplotlib.pyplot as plt

def plot_loss(history):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    # Convertir cada elemento si es cupy.ndarray
    train_loss = [loss.get() if hasattr(loss, "get") else loss for loss in train_loss]
    val_loss = [loss.get() if hasattr(loss, "get") else loss for loss in val_loss]

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train Loss",color='cadetblue')
    plt.plot(val_loss, label="Validation Loss",color='indianred')
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.show




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

def exponential_schedule(initial_lr, gamma=0.995):
    def scheduler(epoch):
        return initial_lr * (gamma ** epoch)
    return scheduler


def linear_schedule(epoch, initial_lr=0.1, final_lr=0.001, saturate_epoch=100):
    if epoch >= saturate_epoch:
        return final_lr
    return initial_lr - (epoch / saturate_epoch) * (initial_lr - final_lr)
