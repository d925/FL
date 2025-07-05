import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes: int = 38):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # 入力画像128×128 → conv+pool×2 → 64チャネル × 32×32 = 65536
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # ← LazyLinearをやめて固定
        self.fc2 = nn.Linear(512, num_classes)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 32, 32)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten (B, 65536)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
