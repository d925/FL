import torch
import torch.nn as nn
import torch.nn.functional as F
from model import CNN
from cluster_emb import MetadataEmbedding

class CNNWithMetadata(nn.Module):
    def __init__(self, num_classes, num_crops, num_diseases, num_regions, emb_dim=32):
        super().__init__()
        self.cnn = CNN(num_classes=num_classes)
        self.metadata_emb = MetadataEmbedding(num_crops, num_diseases, num_regions, emb_dim)
        # trainableに変更
        for p in self.metadata_emb.parameters():
            p.requires_grad = True

        # CNN出力512 + メタ埋め込み (32×3=96) → 結合後に最終分類
        self.fc_fusion = nn.Linear(512 + emb_dim * 3, num_classes)

    def forward(self, x, crop_id, disease_id, region_id):
        cnn_feat = self.cnn._forward_conv(x)
        cnn_feat = cnn_feat.view(cnn_feat.size(0), -1)
        cnn_feat = F.relu(self.cnn.fc1(cnn_feat))
        meta_feat = self.metadata_emb(crop_id, disease_id, region_id)
        fused = torch.cat([cnn_feat, meta_feat], dim=1)
        out = self.fc_fusion(fused)
        return out
