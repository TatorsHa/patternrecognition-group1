# %% [markdown]
# # Exercise 2 – SVM, MLP & CNN on MNIST (TSV loader)
#
# This notebook implements three models on the MNIST dataset using a unified TSV-based data loader:
# - **SVM (Support Vector Machine)**
# - **MLP (Multilayer Perceptron)**
# - **CNN (Convolutional Neural Network)**
#
# The code is structured so that:
# - The **data loading cells** could be run once and reused by all models.
# - It can be started from any model section (SVM / MLP / CNN) independently.

# %%
# ## 1. Imports and setup

import numpy as np  # For numerical operations
import pandas as pd  # For reading TSV files
import torch  # Deep learning framework for MLP/CNN
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Activation functions, loss functions
from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
)  # Dataset and DataLoader utilities
from sklearn.model_selection import StratifiedShuffleSplit  # Splitting tool
from sklearn.metrics import accuracy_score, confusion_matrix  # Evaluation metrics
from pathlib import Path  # For file paths
from PIL import Image  # To load images from disk
import matplotlib.pyplot as plt  # For plotting
import random  # For reproducibility
import json  # For saving results
import os
from datetime import datetime

# %%
# ## 2. Configuration and reproducibility


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)  # Set all seeds

# Define configuration dictionary
cfg = {
    "data_root": "./data/MNIST",  # Path to dataset root containing TSVs
    "train_tsv": "gt-train.tsv",
    "test_tsv": "gt-test.tsv",
    "batch_size": 128,
    "epochs": 10,
    "lr": 1e-3,
    "val_size": 0.1,
    "seed": 42,
}

# Detect device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
# ## 3. Dataset class


class MNISTTSVDataset(Dataset):
    """Custom dataset to load MNIST images listed in TSV files."""

    def __init__(self, tsv_path, data_root, transform=None, as_flat=False):
        self.tsv_path = Path(tsv_path)
        self.data_root = Path(data_root)
        self.transform = transform
        self.as_flat = as_flat
        self.df = pd.read_csv(
            self.tsv_path, sep="\t", header=None, names=["rel_path", "label"]
        )
        self.paths = self.df["rel_path"].tolist()
        self.labels = self.df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load image from relative path
        img_path = self.tsv_path.parent / self.paths[idx]
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        arr = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
        tensor = torch.from_numpy(arr).unsqueeze(0)  # Add channel dimension
        if self.as_flat:
            tensor = tensor.view(-1)  # Flatten to vector for SVM/MLP
        return tensor, self.labels[idx]


# %%
# ## 4. Quick data check


# Small function to visualize examples
def show_samples(dataset, n=5):
    plt.figure(figsize=(n * 2, 2))
    for i in range(n):
        img, label = dataset[i]
        plt.subplot(1, n, i + 1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.show()


# Example usage (uncomment if dataset available):
# sample_ds = MNISTTSVDataset(Path(cfg['data_root'])/cfg['train_tsv'], cfg['data_root'])
# print(sample_ds.df.shape)
# print(sample_ds.df.head())
# show_samples(sample_ds)

# %%
# ## 5. Data splitting and DataLoader creation


# Function to create stratified train/val split
def stratified_split(labels, val_size=0.1, seed=42):
    idx = np.arange(len(labels))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss.split(idx, labels))
    return train_idx, val_idx


# Function to create PyTorch DataLoaders for all splits
def make_loaders(cfg, as_flat=False):
    train_tsv = Path(cfg["data_root"]) / cfg["train_tsv"]
    full_ds = MNISTTSVDataset(train_tsv, cfg["data_root"], as_flat=as_flat)
    train_idx, val_idx = stratified_split(
        full_ds.labels, cfg["val_size"], seed=cfg.get("seed", 42)
    )

    # Subset datasets
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    # Test dataset
    test_tsv = Path(cfg["data_root"]) / cfg["test_tsv"]
    test_ds = MNISTTSVDataset(test_tsv, cfg["data_root"], as_flat=as_flat)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader


# %% [markdown]
# (DEL) The data-loading cells above can be run once and reused for all models below (at least, that was my original idea)
# # 6. Exercise 2a – SVM
# # 7. Exercise 2b – MLP
# %% [markdown]
# # 8. Exercise 2c – CNN
# This section trains a CNN only. It:
# - Defines a simple configurable CNN (`SimpleCNN`)
# - Trains with per-epoch **loss & accuracy** (train/val)
# - Uses **early stopping** on best validation accuracy
# - Saves artifacts under `./runs/cnn/`:
#   - `loss.png`, `acc.png`, `best_cnn.pt`, `predictions.csv`, `confusion_matrix.png`, `results.json`

# %%
# ## 8.1 Model definition


class SimpleCNN(nn.Module):
    def __init__(
        self, in_ch=1, conv_layers=2, kernel_size=5, num_classes=10, dropout=0.25
    ):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(conv_layers):
            out_ch = 32 * (2**i)
            layers.append(nn.Conv2d(ch, out_ch, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(ch * (28 // (2**conv_layers)) ** 2, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


print(SimpleCNN())

# %%
# ## 8.2 Training utilities (metrics, loop, early stopping)


def _epoch_run(model, loader, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * yb.size(0)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return total_loss / total, total_correct / total


# %%
# ## 8.3 Train CNN with early stopping and save artifacts


def train_cnn(cfg, conv_layers=2, kernel_size=5, lr=None, epochs=None, patience=5):
    print("Training CNN...")
    set_seed(cfg.get("seed", 42))
    section_dir = Path("runs") / cfg.get("run_name", "cnn")
    section_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(cfg, as_flat=False)
    model = SimpleCNN(conv_layers=conv_layers, kernel_size=kernel_size).to(device)
    lr = cfg["lr"] if lr is None else lr
    epochs = cfg["epochs"] if epochs is None else epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch = -1.0, -1
    best_path = section_dir / "best_cnn.pt"
    es_count = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _epoch_run(
            model, train_loader, optimizer=optimizer, device=device
        )
        va_loss, va_acc = _epoch_run(model, val_loader, optimizer=None, device=device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(
            f"[{epoch:03d}/{epochs}] "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} acc={va_acc:.4f}"
        )

        # Early stopping (by best val acc)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            es_count = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "conv_layers": conv_layers,
                    "kernel_size": kernel_size,
                    "epochs_trained": epoch,
                    "best_val_acc": float(best_val_acc),
                },
                best_path,
            )
        else:
            es_count += 1
            if es_count >= patience:
                print(
                    f"Early stopping at epoch {epoch} (best at {best_epoch}, acc={best_val_acc:.4f})"
                )
                break

    # Save curves
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("CNN Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(section_dir / "loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("CNN Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(section_dir / "acc.png", dpi=150)
    plt.close()

    # Save training history
    with open(section_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(
        f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch} | checkpoint: {best_path}"
    )
    return best_path, best_val_acc


# %%
# ## 8.4 Final test evaluation (load best checkpoint, evaluate, save CM & predictions)


def evaluate_cnn(cfg, checkpoint_path):
    from sklearn.metrics import ConfusionMatrixDisplay
    import pandas as pd

    section_dir = Path("runs") / cfg.get("run_name", "cnn")
    section_dir.mkdir(parents=True, exist_ok=True)

    # Build loaders (test used only here)
    _, _, test_loader = make_loaders(cfg, as_flat=False)

    # Rebuild model consistent with checkpoint meta if present
    ckpt = torch.load(checkpoint_path, map_location=device)
    conv_layers = ckpt.get("conv_layers", 2)
    kernel_size = ckpt.get("kernel_size", 5)
    model = SimpleCNN(conv_layers=conv_layers, kernel_size=kernel_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    test_acc = accuracy_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    plt.title(f"CNN - Test Confusion Matrix | acc={test_acc:.4f}")
    plt.tight_layout()
    plt.savefig(section_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Predictions CSV
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        section_dir / "predictions.csv", index=False
    )

    # Summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "best_val_acc": float(ckpt.get("best_val_acc", -1.0)),
        "test_acc": float(test_acc),
        "artifacts": {
            "loss_png": str((section_dir / "loss.png").resolve()),
            "acc_png": str((section_dir / "acc.png").resolve()),
            "confusion_matrix_png": str(
                (section_dir / "confusion_matrix.png").resolve()
            ),
            "predictions_csv": str((section_dir / "predictions.csv").resolve()),
            "checkpoint": str((section_dir / "best_cnn.pt").resolve()),
        },
        "cfg": cfg,
    }
    with open(section_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Artifacts saved to: {section_dir.resolve()}")
    return test_acc


# %%
# ## 8.5 Run (train + evaluate)
# Uncomment to run end-to-end:
# best_path, best_val = train_cnn(cfg, conv_layers=2, kernel_size=5, patience=0)
# test_acc = evaluate_cnn(cfg, best_path)

# %%
