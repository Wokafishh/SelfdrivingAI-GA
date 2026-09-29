"""
Behaviour Cloning (BC) Training Module
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(str(Path(__file__).resolve().parent.parent))
from arkitektur import DrivingNetwork


def setup(lr: float = 1e-3):
    """
    Sets up the model, optimizer, data, and single-iteration step function.
    `run_training.py` handles loops, disk saving, Ctrl+C, and resuming.
    """
    torch.manual_seed(42)
    images = torch.randn(3, 3, 64, 64)
    distances = torch.tensor([[0.2], [1.5], [5.0]])

    expert_left = torch.tensor([[-1.0], [0.5], [1.0]])
    expert_right = torch.tensor([[1.0], [1.0], [1.0]])

    model = DrivingNetwork(in_channels=3)

    # Disable BatchNorm running stats for batch size < 4 to prevent gradient death
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    def train_step(epoch: int) -> float:
        """Executes a single training iteration and returns the step loss."""
        optimizer.zero_grad()
        pred_left, pred_right = model(images, distances)

        loss = criterion(pred_left, expert_left) + criterion(pred_right, expert_right)
        loss.backward()
        optimizer.step()

        return loss.item()

    return model, train_step, "Behaviour Cloning (BC)"
