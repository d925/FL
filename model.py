import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import num_labels

class CNN(nn.Module):
    """
    ResNet-18ベースの分類モデル（ImageNet事前学習済み）
    特徴抽出時は最終全結合層直前の512次元を使用
    """
    def __init__(self, num_classes: int = num_labels, pretrained: bool = True):
        super(CNN, self).__init__()
        
        # ImageNet事前学習済みResNet-18をロード
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # 最終全結合層を置き換え（38クラス分類用）
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)
    
    def extract_features(self, x):
        """
        最終全結合層直前の特徴量（512次元）を抽出
        """
        # ResNet-18の構造に従って特徴抽出
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # (batch, 512)
        return x

