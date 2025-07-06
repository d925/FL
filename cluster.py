import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision.models import resnet18
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

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

    features = np.concatenate(features, axis=0)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    return np.concatenate([mean, std])  # 平均＋標準偏差（情報強化）


def determine_optimal_k_auto(features, k_range=(2, 10), plot=False):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # PCAによる次元削減（95%の分散を保持）
    pca_full = PCA()
    pca_full.fit(scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.searchsorted(cum_var, 0.95) + 1

    pca = PCA(n_components=min(n_components, 50))
    reduced = pca.fit_transform(scaled)

    silhouettes = []
    wcss = []
    k_range = range(k_range[0], min(k_range[1] + 1, len(features)))  # num_clients超えを防ぐ

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(reduced)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(reduced, cluster_ids)
        silhouettes.append(score)

    best_k = k_range[np.argmax(silhouettes)]

    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(k_range, wcss, marker='o')
        plt.xlabel('k')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouettes, marker='o', color='orange')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    print(f"🟢 自動判定された最適クラスタ数 k: {best_k}")
    return best_k, scaler, pca


def cluster_clients(num_clients, feature_extractor=None, k_range=(2, 10), plot=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature_extractor is None:
        model = resnet18(pretrained=True)
        model.fc = nn.Identity()
    else:
        model = feature_extractor
    model.to(device)

    client_features = []
    for cid in range(num_clients):
        feat = extract_features(cid, model, device)
        client_features.append(feat)

    client_features = np.vstack(client_features)

    optimal_k, scaler, pca = determine_optimal_k_auto(client_features, k_range, plot)

    scaled = scaler.transform(client_features)
    reduced = pca.transform(scaled)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(reduced)

    print(f"🔍 決定されたクラスタ数: {optimal_k}")
    return {cid: int(cluster_ids[cid]) for cid in range(num_clients)}
