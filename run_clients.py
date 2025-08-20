import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import get_partitioned_data
from config import num_clients
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    return np.concatenate(features, axis=0).mean(axis=0)

def preprocess_features(features, use_pca=True, n_components=50):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    if use_pca:
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(scaled)
        return reduced
    else:
        return scaled

def visualize_clusters(features, cluster_ids):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    for cluster in np.unique(cluster_ids):
        idx = cluster_ids == cluster
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f'Cluster {cluster}', alpha=0.7)
    plt.legend()
    plt.title("Client Feature Clusters (t-SNE 2D Projection)")
    plt.xlabel("TSNE Dim 1")
    plt.ylabel("TSNE Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cluster_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

def determine_optimal_k(features, k_range=range(2, 11)):
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        scores.append(score)
    best_k = k_range[np.argmax(scores)]
    print(f"🧠 最適なクラスタ数: {best_k}, シルエットスコア: {max(scores):.4f}")
    return best_k

def cluster_clients(num_clients, feature_extractor=None, use_pca=True, pca_components=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if feature_extractor is None:
        from model import CNN
        model = CNN(num_classes=38)  # 38クラスの植物病害画像
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

    # KMeansクラスタリング
    best_k = determine_optimal_k(processed_features)
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    cluster_ids = kmeans.fit_predict(processed_features)

    score = silhouette_score(processed_features, cluster_ids)
    print(f"🔍 クラスタ数: {best_k}, シルエットスコア: {score:.4f}")

    visualize_clusters(processed_features, cluster_ids)

    return {cid: int(cluster_ids[cid]) for cid in range(num_clients)}