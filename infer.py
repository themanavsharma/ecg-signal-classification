# infer.py
import numpy as np
import torch
from src.dataset import load_ucr_ts, to_binary_labels
from src.model import ECG1DCNN

def get_device():
    # Apple GPU if available
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main(idx=0):
    device = get_device()

    # load trained checkpoint
    ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)

    # saved train-set normalization stats
    mean = ckpt["scaler_mean"]
    std = ckpt["scaler_std"]

    # load test data + labels
    X, y = load_ucr_ts("data/ECG5000_TEST.ts")
    y_bin = to_binary_labels(y)  # same mapping used in training

    # grab one sample by index + standardize it
    x = (X[idx] - mean) / (std + 1e-8)

    # model expects (batch, channels, length)
    x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,140)

    # rebuild model + load weights
    model = ECG1DCNN(n_classes=2)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # forward pass only
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # 2 probs

    print(
        f"True label (binary): {int(y_bin[idx])}  |  "
        f"P(normal)= {probs[0]:.3f}  P(abnormal)= {probs[1]:.3f}"
    )

if __name__ == "__main__":
    main(idx=1)
