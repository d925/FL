import torch
import torch.nn as nn
import torch.nn.functional as F

class PMACNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # === モジュール1: kernel 4x4, 3x3 + 分岐 ===
        self.module1_conv1 = nn.Conv2d(3, 32, kernel_size=4, padding=1)
        self.module1_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.module1_maxpool1 = nn.MaxPool2d(2)
        self.module1_maxpool2 = nn.MaxPool2d(2)
        self.module1_conv3 = nn.Conv2d(64, 128, kernel_size=1)

        # === モジュール2: kernel 5x5, 2x2 + 分岐 ===
        self.module2_conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.module2_conv2 = nn.Conv2d(32, 64, kernel_size=2, padding=0)
        self.module2_maxpool1 = nn.MaxPool2d(2)
        self.module2_maxpool2 = nn.MaxPool2d(2)
        self.module2_conv3 = nn.Conv2d(64, 128, kernel_size=1)

        # === Attentionモジュール ===
        self.attention_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.attention_maxpool = nn.MaxPool2d(2)
        self.attention_avgpool = nn.AvgPool2d(2)

        # アテンション後のconv（論文の図だとさらにconv層ありそうなので追加）
        self.attention_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # 最終層
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # --- モジュール1 ---
        m1 = F.relu(self.module1_conv1(x))
        m1 = self.module1_maxpool1(m1)
        m1_conv2 = F.relu(self.module1_conv2(m1))
        m1_pool2 = self.module1_maxpool2(m1_conv2)

        # m1_conv2 から並列に3番目convとmaxpoolに入る
        m1_conv3 = F.relu(self.module1_conv3(m1_conv2))
        m1_pool3 = self.module1_maxpool2(m1_conv3)

        # m1の2番目conv層出力と3番目conv層後maxpoolをconcat
        m1_concat = torch.cat([m1_pool2, m1_pool3], dim=1)  # 64+128=192チャネル

        # --- モジュール2 ---
        m2 = F.relu(self.module2_conv1(x))
        m2 = self.module2_maxpool1(m2)
        m2_conv2 = F.relu(self.module2_conv2(m2))
        m2_pool2 = self.module2_maxpool2(m2_conv2)

        # m2_conv2 から並列に3番目convとmaxpoolに入る
        m2_conv3 = F.relu(self.module2_conv3(m2_conv2))
        m2_pool3 = self.module2_maxpool2(m2_conv3)

        # m2の2番目conv層出力と3番目conv層後maxpoolをconcat
        m2_concat = torch.cat([m2_pool2, m2_pool3], dim=1)  # 64+128=192チャネル

        # --- 両モジュールconcat ---
        combined = torch.cat([m1_concat, m2_concat], dim=1)  # 192 + 192 = 384チャネル

        # --- Attentionモジュール ---
        att = F.relu(self.attention_conv(combined))  # 384→128チャネル

        att_max = self.attention_maxpool(att)
        att_avg = self.attention_avgpool(att)
        att_cat = torch.cat([att_max, att_avg], dim=1)  # 128*2=256チャネル

        # 元のattの空間サイズにアップサンプリング（bilinear）
        att_cat_upsampled = F.interpolate(att_cat, size=att.shape[2:], mode='bilinear', align_corners=False)

        # att と att_cat_upsampledのチャネルは異なるのでatt_cat_upsampledをチャネル半分だけ使い掛け算
        att_weighted = att * att_cat_upsampled[:, :att.size(1), :, :]

        # さらにconv層で調整
        att_weighted = F.relu(self.attention_conv2(att_weighted))

        # --- グローバルプーリング、全結合 ---
        pooled = self.global_maxpool(att_weighted)  # (B,128,1,1)
        pooled = pooled.view(pooled.size(0), -1)  # (B,128)
        pooled = self.dropout(pooled)
        out = self.fc(pooled)  # (B,num_classes)

        return out
