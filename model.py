# model.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 38):
        super(MobileNetClassifier, self).__init__()
        base_model = mobilenet_v2(pretrained=True)  # 事前学習ありにしておくと性能向上
        self.feature_extractor = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
