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


def train(epochs: int = 100, lr: float = 1e-3) -> dict:
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

    start_loss = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_left, pred_right = model(images, distances)

        loss = criterion(pred_left, expert_left) + criterion(pred_right, expert_right)

        if epoch == 0:
            start_loss = loss.item()

        loss.backward()
        optimizer.step()

    return {
        "name": "Behaviour Cloning (BC)",
        "iterations": epochs,
        "start_loss": start_loss,
        "final_loss": loss.item()
    }
