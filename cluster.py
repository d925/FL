from torchvision.models import resnet18
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import torch
from torch.utils.data import DataLoader
import numpy as np
from utils import get_partitioned_data
from config import num_clients

def extract_features(client_id, model, device):
    dataset, _ = get_partitioned_data(client_id, num_clients)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    features = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            feat = model(x)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)

def cluster_clients(num_clients, num_clusters, feature_extractor=None, pca_components=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if feature_extractor is None:
        model = resnet18(pretrained=True)
        model.fc = nn.Identity()  # 最終層を除去して特徴抽出器に
    else:
        model = feature_extractor
    model.to(device)

    all_features = []
    for cid in range(num_clients):
        feat = extract_features(cid, model, device)
        # clientごとに特徴の平均だけでなく、全特徴をまとめて使うために後でPCAを使う
        all_features.append(feat)

    # クライアント単位の特徴ベクトル（平均）を作る
    client_features = [np.mean(feat, axis=0) for feat in all_features]
    client_features = np.stack(client_features)

    # PCAで次元削減（高次元空間だとクラスタリングが辛いので）
    pca = PCA(n_components=pca_components, random_state=42)
    reduced_features = pca.fit_transform(client_features)

    # GMMクラスタリング（KMeansより柔軟に分布を捉えられる）
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    cluster_ids = gmm.fit_predict(reduced_features)

    return {cid: int(cluster_ids[cid]) for cid in range(num_clients)}
