import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 38):
        super(MobileNetClassifier, self).__init__()
        # 事前学習済みのMobileNetV2をロード（特徴抽出器として使用）
        base_model = mobilenet_v2(pretrained=False)

        # 分類ヘッドの入れ替え
        self.feature_extractor = base_model.features  # Convブロック群
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # グローバル平均プーリング
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),  # MobileNetV2の最終出力チャンネルは1280
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # (B, 1280, H, W)
        x = self.avgpool(x)            # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)        # (B, 1280)
        x = self.classifier(x)         # (B, num_classes)
        return x
