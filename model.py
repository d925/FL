import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

class MobileNetV3Classifier(nn.Module):
    def __init__(self, num_classes: int = 38):
        super(MobileNetV3Classifier, self).__init__()
        # 事前学習済みのMobileNetV3 Largeを特徴抽出器としてロード
        base_model = mobilenet_v3_large(pretrained=False)

        # 特徴抽出部分（features）と分類ヘッドを分ける
        self.feature_extractor = base_model.features  # Convブロック群
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # グローバル平均プーリング

        # V3 Largeの最後のconv出力チャンネルは1280（V2と同じ）
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # (B, 1280, H, W)
        x = self.avgpool(x)            # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)        # (B, 1280)
        x = self.classifier(x)         # (B, num_classes)
        return x
