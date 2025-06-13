import torch
import torch.nn as nn
import torch.nn.functional as F

class PMACNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 親子で並列な2つのスケールモジュールを用意
        # モジュール1: カーネル4x4, 3x3
        self.module1_conv1 = nn.Conv2d(3, 32, kernel_size=4, padding=1)
        self.module1_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.module1_maxpool1 = nn.MaxPool2d(2)
        self.module1_maxpool2 = nn.MaxPool2d(2)
        self.module1_conv3 = nn.Conv2d(64, 128, kernel_size=1)  # 1x1 conv
        
        # モジュール2: カーネル5x5, 2x2
        self.module2_conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.module2_conv2 = nn.Conv2d(32, 64, kernel_size=2, padding=0)
        self.module2_maxpool1 = nn.MaxPool2d(2)
        self.module2_maxpool2 = nn.MaxPool2d(2)
        self.module2_conv3 = nn.Conv2d(64, 128, kernel_size=1)  # 1x1 conv

        # アテンション用の畳み込み層とプーリング層
        self.attention_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.attention_maxpool = nn.MaxPool2d(2)
        self.attention_avgpool = nn.AvgPool2d(2)

        # 最終分類用全結合層
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # モジュール1処理
        m1 = F.relu(self.module1_conv1(x))
        m1 = self.module1_maxpool1(m1)
        m1 = F.relu(self.module1_conv2(m1))
        m1 = self.module1_maxpool2(m1)
        m1_conv3 = F.relu(self.module1_conv3(m1))
        m1_pool = self.module1_maxpool2(m1_conv3)

        # モジュール2処理
        m2 = F.relu(self.module2_conv1(x))
        m2 = self.module2_maxpool1(m2)
        m2 = F.relu(self.module2_conv2(m2))
        m2 = self.module2_maxpool2(m2)
        m2_conv3 = F.relu(self.module2_conv3(m2))
        m2_pool = self.module2_maxpool2(m2_conv3)

        # 両モジュールの連結
        concat = torch.cat([m1_pool, m2_pool], dim=1)  # チャンネル方向に連結 (128+128=256)

        # アテンションモジュール
        att = F.relu(self.attention_conv(concat))
        att_max = self.attention_maxpool(att)
        att_avg = self.attention_avgpool(att)
        att_cat = torch.cat([att_max, att_avg], dim=1)  # チャンネル方向に連結

        # アテンションで強調
        att_cat_upsampled = F.interpolate(att_cat, size=att.shape[2:], mode='bilinear', align_corners=False)
        att_mul = att * att_cat_upsampled[:, :att.size(1), :, :]  # チャンネル数を合わせて乗算

        # グローバルプーリング＋分類
        pooled = self.global_maxpool(att_mul)
        pooled = pooled.view(pooled.size(0), -1)
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        return out
