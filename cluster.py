import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import get_partitioned_data
from config import num_clients
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from sklearn.cluster import DBSCAN

def determine_optimal_k(features, eps_range=np.linspace(0.1, 5.0, 50), min_samples=5):
    best_eps = None
    best_score = -1
    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters <= 1:
            continue
        score = silhouette_score(features[labels != -1], labels[labels != -1])
        if score > best_score:
            best_score = score
            best_eps = eps
    if best_eps is None:
        print("⚠️ 有効なepsが見つかりませんでした。DBSCANクラスタリングは失敗しました。")
        # とりあえずeps=0.5で実行させる
        best_eps = 0.5
        best_score = -1
    print(f"🧠 最適なeps: {best_eps}, シルエットスコア: {best_score:.4f}")
    return best_eps

def cluster_clients(num_clients, feature_extractor=None, use_pca=True, pca_components=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature_extractor is None:
        from model import CNN
        model = CNN(num_classes=128)
        model.fc2 = nn.Identity()
    else:
        model = feature_extractor
    model.to(device)

    client_features = []
    for cid in range(num_clients):
        feat = extract_features(cid, model, device)
        client_features.append(feat)
        if (cid + 1) % 10 == 0:
            torch.cuda.empty_cache()

    client_features = np.vstack(client_features)
    model.cpu()
    torch.cuda.empty_cache()

    processed_features = preprocess_features(client_features, use_pca=use_pca, n_components=pca_components)

    # DBSCANのパラメータチューニング
    best_eps = determine_optimal_k(processed_features)

    dbscan = DBSCAN(eps=best_eps, min_samples=5)
    cluster_ids = dbscan.fit_predict(processed_features)

    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    noise_points = np.sum(cluster_ids == -1)
    print(f"🔍 クラスタ数: {n_clusters}, ノイズ点数: {noise_points}")

    if n_clusters > 1:
        score = silhouette_score(processed_features[cluster_ids != -1], cluster_ids[cluster_ids != -1])
        print(f"🔎 シルエットスコア（DBSCAN）: {score:.4f}")
    else:
        print("⚠️ クラスタが1つ以下のためシルエットスコア計算不可")

    visualize_clusters(processed_features, cluster_ids)

    return {cid: int(cluster_ids[cid]) for cid in range(num_clients)}
