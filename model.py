import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 38):
        super(ResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=False)  # 必要なら pretrained=True に
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 分類層の置き換え（出力ユニット数調整）
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
