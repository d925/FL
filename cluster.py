from torchvision.models import resnet18
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
    return np.concatenate(features, axis=0).mean(axis=0)  # クライアントごとに平均プール

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

def determine_optimal_k_auto(features, k_range=(2, 15), plot=False):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # 次元削減（PCA）
    pca = PCA(n_components=min(50, scaled.shape[1]))
    reduced = pca.fit_transform(scaled)

    silhouettes = []
    wcss = []

    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_ids = kmeans.fit_predict(reduced)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(reduced, cluster_ids)
        silhouettes.append(score)

    # シルエット最大のkを選択（より信頼性が高い指標）
    best_k = np.argmax(silhouettes) + k_range[0]

    if plot:
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(range(k_range[0], k_range[1] + 1), wcss, marker='o')
        plt.xlabel('Number of clusters k')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')

        plt.subplot(1,2,2)
        plt.plot(range(k_range[0], k_range[1] + 1), silhouettes, marker='o', color='orange')
        plt.xlabel('Number of clusters k')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')

        plt.show()

    print(f"🟢 自動判定された最適クラスタ数 k: {best_k}")
    return best_k, scaler, pca

def cluster_clients_improved(num_clients, feature_extractor=None, k_range=(2, 15), plot=False):
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

    # 自動で最適kを判定し、スケーラーとPCAも取得
    optimal_k, scaler, pca = determine_optimal_k_auto(client_features, k_range, plot)

    # 特徴量の前処理（同じ処理をクラスタリングに使う）
    scaled = scaler.transform(client_features)
    reduced = pca.transform(scaled)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    cluster_ids = kmeans.fit_predict(reduced)

    print(f"🔍 決定されたクラスタ数: {optimal_k}")
    return {cid: int(cluster_ids[cid]) for cid in range(num_clients)}
