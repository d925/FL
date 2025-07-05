import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

class MobileNetV3Classifier(nn.Module):
    def __init__(self, num_classes: int = 38):
        super(MobileNetV3Classifier, self).__init__()
        base_model = mobilenet_v3_large(pretrained=False)
        self.feature_extractor = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 🔍 入力チャンネル数をダミーデータで自動取得
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.feature_extractor(dummy_input)
            self.out_features = dummy_output.shape[1]  # たとえば960とか

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.out_features, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
