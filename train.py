# train.py
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import load_ucr_ts, to_binary_labels, Standardizer, ECGDataset
from src.model import ECG1DCNN


def get_device():
    # Use Apple GPU (MPS) if available, otherwise CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accuracy(logits, y):
    # logits: [batch, n_classes] -> predicted class index
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def main():
    device = get_device()
    print("Device:", device)

    X, y = load_ucr_ts("data/ECG5000_TRAIN.ts")
    y = to_binary_labels(y)

    # Keep class balance similar in train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit normalization on train only
    scaler = Standardizer.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_val = scaler.transform(X_val)

    train_ds = ECGDataset(X_tr, y_tr)
    val_ds = ECGDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = ECG1DCNN(n_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val = 0.0
    Path("checkpoints").mkdir(exist_ok=True)

    for epoch in range(1, 31):
        model.train()
        tr_losses, tr_accs = [], []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            tr_losses.append(loss.item())
            tr_accs.append(accuracy(logits, yb))

        model.eval()
        val_accs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_accs.append(accuracy(logits, yb))
        val_acc = float(np.mean(val_accs))

        print(
            f"Epoch {epoch:02d} | loss={np.mean(tr_losses):.4f} | "
            f"tr_acc={np.mean(tr_accs):.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "scaler_mean": scaler.mean,
                    "scaler_std": scaler.std,
                },
                "checkpoints/best.pt",
            )

    print("Best val acc:", best_val)


if __name__ == "__main__":
    main()
