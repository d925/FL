# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import num_labels

# Optional: pretrained ResNet backbone
try:
    from torchvision import models as tv_models
    _TORCHVISION_AVAILABLE = True
except Exception:
    _TORCHVISION_AVAILABLE = False

class CNN(nn.Module):
    """
    Improved lightweight CNN with BatchNorm and an option to use a pretrained ResNet18 backbone.

    Args:
        num_classes: number of output classes (from config.num_labels)
        backbone: 'small' (default) or 'resnet18'
        pretrained: if backbone == 'resnet18', load ImageNet pretrained weights (bool)
    """
    def __init__(self, num_classes: int = num_labels, backbone: str = "resnet18", pretrained: bool = False):
        super(CNN, self).__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            if not _TORCHVISION_AVAILABLE:
                raise RuntimeError("torchvision is required for backbone='resnet18'. Install torchvision.")
            # Load torchvision resnet18
            resnet = tv_models.resnet18(pretrained=pretrained)
            # Remove the original fc and use it as feature extractor.
            # Keep batchnorm behavior: running stats are buffers local to the module,
            # which we will *not* explicitly synchronize (FedBN-style).
            modules = list(resnet.children())[:-1]  # remove the final FC
            self.feature_extractor = nn.Sequential(*modules)  # outputs (B, 512, 1, 1)
            feat_dim = 512
            # small projection head
            self.fc_proj = nn.Linear(feat_dim, 256)
            self.dropout = nn.Dropout(0.5)
            self.classifier = nn.Linear(256, num_classes)

            # initialize classifier if needed
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0.0)

        else:
            # small convolutional backbone (improved)
            # Input 128x128 -> conv1 + pool -> 64x64
            # -> conv2 + pool -> 32x32
            # -> conv3 + pool -> 16x16
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.25)

            # After three poolings: 128 x 16 x 16 -> flatten dim = 128*16*16 = 32768
            # Reduce fc size to avoid huge parameter count
            self.fc1 = nn.Linear(128 * 16 * 16, 1024)
            self.fc2 = nn.Linear(1024, num_classes)

            # Initialize
            nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
            nn.init.normal_(self.fc1.weight, 0, 0.01)
            nn.init.normal_(self.fc2.weight, 0, 0.01)
            if self.fc1.bias is not None:
                nn.init.constant_(self.fc1.bias, 0.0)
            if self.fc2.bias is not None:
                nn.init.constant_(self.fc2.bias, 0.0)

    def _forward_small(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B,32,64,64)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B,64,32,32)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (B,128,16,16)
        return x

    def forward(self, x):
        if self.backbone_name == "resnet18":
            # Expect input maybe 224x224; if input smaller, upsample externally
            feats = self.feature_extractor(x)  # (B, 512, 1, 1)
            feats = feats.view(feats.size(0), -1)  # (B, 512)
            z = F.relu(self.fc_proj(feats))
            z = self.dropout(z)
            out = self.classifier(z)
            return out
        else:
            x = self._forward_small(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
