# evaluate.py
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

from src.dataset import load_ucr_ts, to_binary_labels, ECGDataset
from src.model import ECG1DCNN

def get_device():
    # use Apple GPU if available
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    device = get_device()

    # load trained checkpoint
    ckpt = torch.load("checkpoints/best.pt", map_location="cpu",weights_only=False)

    # saved normalization stats from training
    mean = ckpt["scaler_mean"]
    std = ckpt["scaler_std"]

    # load test split
    X_test, y_test = load_ucr_ts("data/ECG5000_TEST.ts")

    # match label setup from training
    y_test = to_binary_labels(y_test)

    # same scaling as training
    X_test = (X_test - mean) / (std + 1e-8)

    # dataset + batching
    ds = ECGDataset(X_test, y_test)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)

    # rebuild model and plug in weights
    model = ECG1DCNN(n_classes=2)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()  # eval mode

    all_probs = []
    all_y = []

    # inference only
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)

            # forward pass -> logits
            logits = model(xb)

            # prob of class 1
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

            all_probs.append(probs)
            all_y.append(yb.numpy())

    # flatten batches
    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_probs)

    # default threshold
    y_pred = (y_prob >= 0.5).astype(int)

    # metrics
    print("AUROC:", roc_auc_score(y_true, y_prob))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    main()
