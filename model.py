import torch
import torch.nn as nn
import torch.nn.functional as F

class PMACNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # === モジュール1 ===
        self.module1_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.module1_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.module1_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.module1_pool2 = nn.MaxPool2d(2)

        # === モジュール2 ===
        self.module2_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.module2_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.module2_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.module2_pool2 = nn.MaxPool2d(2)

        # === Attention層 ===
        self.attention_conv = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.attention_maxpool = nn.MaxPool2d(2)
        self.attention_avgpool = nn.AvgPool2d(2)

        # チャネル整合用の1x1 conv
        self.attention_channel_match = nn.Conv2d(256, 128, kernel_size=1)

        # attentionの後のconv
        self.attention_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # --- モジュール1 ---
        m1 = self.module1_conv1(x)
        m1_conv2 = self.module1_conv2(m1)
        m1_pool2 = self.module1_pool2(m1_conv2)
        m1_conv3 = self.module1_conv3(m1_conv2)
        m1_pool3 = self.module1_pool2(m1_conv3)
        m1_concat = torch.cat([m1_pool2, m1_pool3], dim=1)  # (B,192,H,W)

        # --- モジュール2 ---
        m2 = self.module2_conv1(x)
        m2_conv2 = self.module2_conv2(m2)
        m2_pool2 = self.module2_pool2(m2_conv2)
        m2_conv3 = self.module2_conv3(m2_conv2)
        m2_pool3 = self.module2_pool2(m2_conv3)
        m2_concat = torch.cat([m2_pool2, m2_pool3], dim=1)  # (B,192,H,W)

        # --- concat ---
        combined = torch.cat([m1_concat, m2_concat], dim=1)  # (B,384,H,W)

        # --- Attention ---
        att = self.attention_conv(combined)
        att_max = self.attention_maxpool(att)
        att_avg = self.attention_avgpool(att)
        att_cat = torch.cat([att_max, att_avg], dim=1)  # (B,256,H/2,W/2)
        att_cat = self.attention_channel_match(att_cat)  # → (B,128,H/2,W/2)

        att_cat_upsampled = F.interpolate(att_cat, size=att.shape[2:], mode='bilinear', align_corners=False)
        att_weighted = att * att_cat_upsampled
        att_weighted = self.attention_conv2(att_weighted)

        pooled = self.global_maxpool(att_weighted).view(att.size(0), -1)
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        return out
