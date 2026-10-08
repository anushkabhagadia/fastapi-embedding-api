import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Assignment 2 architecture: 64x64x3 input -> 10 classes."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)   # 64x64x16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                  # halves H and W
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 32x32x32
        # after conv1+pool: 32x32x16 ; after conv2+pool: 16x16x32
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (B,16,32,32)
        x = self.pool(F.relu(self.conv2(x)))   # (B,32,16,16)
        x = x.flatten(start_dim=1)             # (B,8192)
        x = F.relu(self.fc1(x))                # (B,100)
        return self.fc2(x)                     # (B,10) raw logits
