import torch
import torch.nn as nn
import torch.nn.functional as F

class PMACNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.module1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.module1_branch = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.module2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.module2_branch = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.attention_reduce = nn.Conv2d(384, 128, kernel_size=3, padding=1)
        self.attention_post = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        m1 = self.module1(x)
        print("m1.shape:", m1.shape)
        print("module1_branch(m1).shape:", self.module1_branch(m1).shape)

        m1_concat = torch.cat([m1, self.module1_branch(m1)], dim=1)

        m2 = self.module2(x)
        m2_concat = torch.cat([m2, self.module2_branch(m2)], dim=1)

        combined = torch.cat([m1_concat, m2_concat], dim=1)
        att = F.relu(self.attention_reduce(combined))

        att_max = F.max_pool2d(att, 2)
        att_avg = F.avg_pool2d(att, 2)
        att_cat = torch.cat([att_max, att_avg], dim=1)
        att_cat_upsampled = F.interpolate(att_cat, size=att.shape[2:], mode='bilinear', align_corners=False)

        att_weighted = att * att_cat_upsampled[:, :att.size(1), :, :]
        att_weighted = F.relu(self.attention_post(att_weighted))

        pooled = self.global_maxpool(att_weighted).view(att.size(0), -1)
        return self.fc(self.dropout(pooled))
