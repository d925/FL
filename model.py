# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int = 38):
        super(ResNet50Classifier, self).__init__()
        base_model = resnet50(pretrained=True)  # ImageNetでの事前学習あり

        # 最終の全結合層（fc）を置き換える
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, num_classes)

        self.model = base_model

    def forward(self, x):
        return self.model(x)
