"""
Neural Network for a Self-Driving AI agent.
Inputs:
  - 1 camera frame (64x64 RGB or Grayscale)
  - Distance sensor reading to closest object

Outputs:
  - left_wheel:  int [-1, 0, 1] (reverse to forward)
  - right_wheel: int [-1, 0, 1] (reverse to forward)
"""

import torch
from torch import nn

# Mapping index [0, 1, 2] -> discrete actions [-1, 0, 1]
ACTIONS = [-1, 0, 1]


class DrivingNetwork(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        # 1. Feature extraction from 64x64 image
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
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
            nn.ReLU(),
        )

        # 3. Output heads producing logits for 3 discrete actions [-1, 0, 1]
        self.left_wheel_head = nn.Linear(512, 3)
        self.right_wheel_head = nn.Linear(512, 3)

    def forward(
        self, image: torch.Tensor, distance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if distance.dim() == 1:
            distance = distance.unsqueeze(1)

        # Extract visual features
        x_img = self.cnn(image)
        x_img = torch.flatten(x_img, start_dim=1)

        # Combine with distance
        x = torch.cat((x_img, distance), dim=1)

        # Dense layers
        x = self.hidden(x)

        # Compute action logits shape: (batch_size, 3)
        left_logits = self.left_wheel_head(x)
        right_logits = self.right_wheel_head(x)

        return left_logits, right_logits

    @torch.no_grad()
    def get_action(
        self, image_tensor: torch.Tensor, distance_val: float
    ) -> tuple[int, int]:
        """Convenience method to pass a single frame + distance float from your
        game/sim loop and return discrete integer actions [-1, 0, 1] for the
        wheels."""
        self.eval()

        # Add batch dimension if passing a single image (C, 64, 64) -> (1, C, 64, 64)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Convert distance scalar float into tensor shape (1, 1) matching device
        dist_tensor = torch.tensor(
            [[distance_val]], dtype=torch.float32, device=image_tensor.device
        )

        # Run inference
        left_logits, right_logits = self.forward(image_tensor, dist_tensor)

        # Pick class index with max probability and map to [-1, 0, 1]
        left_idx = torch.argmax(left_logits, dim=1).item()
        right_idx = torch.argmax(right_logits, dim=1).item()

        return ACTIONS[left_idx], ACTIONS[right_idx]


# Test execution
if __name__ == "__main__":
    model = DrivingNetwork(in_channels=3)

    # Simulate 1 frame from game/sensor
    dummy_frame = torch.randn(3, 64, 64)  # (C, H, W)
    dummy_distance = 2.4  # Distance to wall in meters/units

    left_action, right_action = model.get_action(dummy_frame, dummy_distance)

    print(f"Left Wheel Action:  {left_action}")  # Integer: -1, 0, or 1
    print(f"Right Wheel Action: {right_action}")  # Integer: -1, 0, or 1
