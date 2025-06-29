from torchvision.models import resnet18
import torch.nn as nn

class ResNetFL(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.base = resnet18(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
    def forward(self, x):
        return self.base(x)
