# src/dataset.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def load_ucr_ts(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a univariate UCR/UEA .ts file where each sample is:
    comma-separated floats then ':' then class label.

    Returns:
        X: (n_samples, series_len) float32
        y: (n_samples,) int64  original labels as ints
    """
    path = Path(path)
    data_started = False
    X_list, y_list = [], []

    # Parse the data section: skip header lines until "@data" marker,
    # then read each line as comma-separated feature values followed by ":label"
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not data_started:
                if line.lower() == "@data":
                    data_started = True
                continue

            # each line: v1,v2,...,v140:label
            series_str, label_str = line.rsplit(":", 1)
            vals = np.array(series_str.split(","), dtype=np.float32)
            y = int(label_str)
            X_list.append(vals)
            y_list.append(y)

    # Stack lists into 2D arrays with proper dtypes matching docstring specs
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def to_binary_labels(y: np.ndarray) -> np.ndarray:
    """
    ECG5000 labels are 1..5. We'll map:
      1 -> 0 (normal)
      2..5 -> 1 (abnormal)
    """
    return (y != 1).astype(np.int64)

@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    
    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        # Compute per-feature mean and std deviation from training data.
        # Add small epsilon (1e-8) to std to avoid division by zero during normalization.
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Apply z-score normalization using fitted mean and std (standardization).
        return (X - self.mean) / self.std

class ECGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # PyTorch conv1d expects (channels, length)
        self.X = torch.from_numpy(X).float().unsqueeze(1)  # (N, 1, L)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
