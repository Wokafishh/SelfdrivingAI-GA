"""
Neural Network for a Self-Driving AI agent (Vision-only version).
Inputs:
  - 1 camera frame (64x64 RGB), float values in [0, 1]

Outputs:
  - left_wheel:  float in [-1, 1] (reverse to forward)
  - right_wheel: float in [-1, 1] (reverse to forward)
"""

import torch
from torch import nn
from PIL import Image
from torchvision.transforms import functional as TF

IMAGE_SIZE = 64


def image_to_uint8_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Resizes a frame to 64x64 RGB and returns a uint8 tensor (3, 64, 64)."""
    img = pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return TF.pil_to_tensor(img)


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """Shared preprocessing for training and inference: float tensor (3, 64, 64) in [0, 1]."""
    return image_to_uint8_tensor(pil_img).float() / 255.0


class DrivingNetwork(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        # 1. Feature extraction from 64x64 image: 64 -> 32 -> 16 -> 8
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Flattened CNN size: 64 channels * 8 * 8 = 4096
        cnn_out_dim = 64 * 8 * 8

        # 2. Small dense head (kept small so it doesn't memorise a small dataset)
        self.hidden = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        # 3. Output heads: one continuous value per wheel, squashed to [-1, 1]
        self.left_wheel_head = nn.Linear(64, 1)
        self.right_wheel_head = nn.Linear(64, 1)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract visual features
        x_img = self.cnn(image)
        x = torch.flatten(x_img, start_dim=1)

        # Dense layers
        x = self.hidden(x)

        # Wheel speeds shape: (batch_size, 1)
        left = torch.tanh(self.left_wheel_head(x))
        right = torch.tanh(self.right_wheel_head(x))

        return left, right

    @torch.no_grad()
    def get_action(
        self, image_tensor: torch.Tensor
    ) -> tuple[float, float]:
        """Convenience method to pass a single frame from your
        game/sim loop and return wheel speeds in [-1, 1]."""
        self.eval()

        # Add batch dimension if passing a single image (C, 64, 64) -> (1, C, 64, 64)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        left, right = self.forward(image_tensor)
        return left[0, 0].item(), right[0, 0].item()


# Test execution
if __name__ == "__main__":
    model = DrivingNetwork(in_channels=3)

    # Simulate 1 frame from game/sensor
    dummy_frame = torch.rand(3, 64, 64)  # (C, H, W)

    left_action, right_action = model.get_action(dummy_frame)

    print(f"Left Wheel Action:  {left_action:+.3f}")
    print(f"Right Wheel Action: {right_action:+.3f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
