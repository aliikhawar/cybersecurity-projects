"""
Neural network for the intrusion detection system.

A small feedforward network that classifies one flow at a time.
Use num_classes=2 for normal vs attack, or a higher number when
each attack type (DoS, Port Scan, Brute Force, and so on) is its own class.
The forward pass returns raw scores (logits), which nn.CrossEntropyLoss expects.
"""

import torch
import torch.nn as nn


class IDSNet(nn.Module):
    """Fully connected network: features -> hidden layers -> class scores."""

    def __init__(self, input_size, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def build_model(input_size, num_classes=2, device=None):
    """Create the network and move it to the GPU when CUDA is available."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IDSNet(input_size, num_classes).to(device)
    return model, device


if __name__ == "__main__":
    # Smoke test: 41 features matches NSL-KDD; 5 classes is normal + 4 attack groups.
    model, device = build_model(input_size=41, num_classes=5)
    sample = torch.randn(8, 41, device=device)
    scores = model(sample)
    print("Device:", device)
    print("Output shape (batch, classes):", tuple(scores.shape))
