# src/model.py
import torch
import torch.nn as nn

class ECG1DCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()

        # Input is expected to be (batch, channels, length) = (B, 1, L) for a single-lead ECG.
        # These conv blocks learn local waveform patterns, then we downsample the time axis a bit.
        self.net = nn.Sequential(
            # Block 1: 1 -> 16 channels, keep length the same, then halve it with pooling
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2: 16 -> 32 channels, same idea
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3: 32 -> 64 channels
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # Collapse the time dimension down to a single value per channel
            nn.AdaptiveAvgPool1d(1),  # (B, 64, 1)
        )

        # Final classifier: map the 64 pooled features to class logits
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        # After the conv stack we have (B, 64, 1); squeeze removes that trailing 1 -> (B, 64)
        x = self.net(x).squeeze(-1)

        # Returns logits (raw scores). For binary classification with CrossEntropyLoss:
        # shape is (B, 2) = [score_normal, score_abnormal]
        return self.head(x)
