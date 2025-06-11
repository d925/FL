import torch
import torch.nn as nn
from torchvision import models

class MobileNetV2_FL(nn.Module):
    def __init__(self, num_classes, freeze_features=True):
        super().__init__()
        # 事前学習済みモデルを読み込み
        base_model = models.mobilenet_v2(weights=None)  # 転移学習したいなら weights='IMAGENET1K_V1'
        
        if freeze_features:
            for param in base_model.features.parameters():
                param.requires_grad = False
        
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base_model.last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
