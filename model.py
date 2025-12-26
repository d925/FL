import torch
import torch.nn as nn
import torch.nn.functional as F
from config import num_labels

class CNN(nn.Module):
    def __init__(self, num_classes: int = num_labels):
        super(CNN, self).__init__()

        # Convolution + GroupNorm
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(4, 32)   # 32ch → 4グループ推奨

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(4, 64)

        self.pool = nn.MaxPool2d(2, 2)

        # Dropout弱め
        self.dropout = nn.Dropout(0.1)

        # FCを縮小（安定性向上）
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))  # (B, 32, 64, 64)
        x = self.pool(F.relu(self.gn2(self.conv2(x))))  # (B, 64, 32, 32)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
