"""
Neural Network for a Self-Driving AI agent.
Inputs:
  - 1 camera frame (64x64 RGB or Grayscale)
  - Distance sensor reading to closest object

Outputs:
  - left_wheel:  float [-1.0 to 1.0] (reverse to forward)
  - right_wheel: float [-1.0 to 1.0] (reverse to forward)
"""

import torch
import torch.nn as nn


class DrivingNetwork(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(DrivingNetwork, self).__init__()

        # 1. Feature extraction from 64x64 image
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Flattened CNN size: 32 channels * 16 * 16 = 8192
        cnn_out_dim = 32 * 16 * 16
        combined_dim = cnn_out_dim + 1  # 8192 image features + 1 distance value

        # 2. 5 hidden layers with 512 neurons each
        self.hidden = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # 3. Output heads for left and right wheels
        self.left_wheel_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Tanh()
        )
        self.right_wheel_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, image: torch.Tensor, distance: torch.Tensor):
        # Extract visual features
        x_img = self.cnn(image)
        x_img = torch.flatten(x_img, start_dim=1)

        # Combine with distance
        x = torch.cat((x_img, distance), dim=1)

        # Dense layers
        x = self.hidden(x)

        # Compute outputs [-1 to 1]
        left_wheel = self.left_wheel_head(x)
        right_wheel = self.right_wheel_head(x)

        return left_wheel, right_wheel

    @torch.no_grad()
    def get_action(self, image_tensor: torch.Tensor, distance_val: float) -> tuple[float, float]:
        """
        Convenience method to pass a single frame + distance float from your game/sim loop
        and return standard Python float actions for the wheels.
        """
        self.eval()

        # Add batch dimension if passing a single image (3, 64, 64) -> (1, 3, 64, 64)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Convert distance scalar float into tensor shape (1, 1)
        dist_tensor = torch.tensor([[distance_val]], dtype=torch.float32)

        # Run inference
        left_out, right_out = self.forward(image_tensor, dist_tensor)

        # Return clean Python floats
        return left_out.item(), right_out.item()

# Litet blaj test
if __name__ == "__main__":
    # Test network initialization
    model = DrivingNetwork(in_channels=3)

    # Simulate 1 frame from game/sensor
    dummy_frame = torch.randn(3, 64, 64)  # (C, H, W)
    dummy_distance = 2.4                 # Distance to wall in meters/units

    left_speed, right_speed = model.get_action(dummy_frame, dummy_distance)

    print(f"Left Wheel Action:  {left_speed:.2f}")   # e.g., -0.74 (reverse)
    print(f"Right Wheel Action: {right_speed:.2f}")  # e.g.,  0.81 (forward)
