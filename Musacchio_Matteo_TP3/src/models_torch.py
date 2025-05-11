import torch
import torch.nn as nn
import torch.nn.functional as F

class MTorchNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.0, use_batchnorm=False):
        super(MTorchNet, self).__init__()
        layers = []
        in_features = input_size

        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_units

        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
from torch.utils.data import DataLoader, TensorDataset
def train_model_torch(X_train, y_train, X_val, y_val, hidden_layers, lr=0.01, batch_size=64,
                   dropout=0.0, use_batchnorm=False, weight_decay=0.0, scheduler_type=None,
                   use_adam=True, epochs=100, patience=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preparar datasets
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = MTorchNet(X_train.shape[1], hidden_layers, y_train.shape[1],
                  dropout=dropout, use_batchnorm=use_batchnorm).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) if use_adam \
        else torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)
    elif scheduler_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        scheduler = None

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb.argmax(dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb.argmax(dim=1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        if patience > 0:
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("⏹ Early stopping triggered.")
                    break

            if scheduler:
                scheduler.step()
        else:
            if scheduler:
                scheduler.step()
        # Guardar el mejor modelo
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()

    # Cargar mejor modelo
    model.load_state_dict(best_model_state)
    return model, history
