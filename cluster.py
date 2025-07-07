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
    # 標準化
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # 次元削減（任意）
    if use_pca:
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(scaled)
        return reduced
    else:
        return scaled


def determine_optimal_k(features, k_range=(2, 10)):
    wcss = []
    silhouettes = []

    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_ids = kmeans.fit_predict(features)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(features, cluster_ids)
        silhouettes.append(score)

    # エルボー法（2階差分最小点）
    deltas = np.diff(wcss)
    elbow_k = np.argmin(np.abs(np.diff(deltas))) + k_range[0] + 1  # +1 for second diff offset

    # シルエット最大
    best_silhouette_k = np.argmax(silhouettes) + k_range[0]

    # どちらか保守的な方を選ぶ
    optimal_k = max(elbow_k, best_silhouette_k)
    print(f"🧠 エルボー法による k: {elbow_k}, シルエット法による k: {best_silhouette_k}, 採用 k: {optimal_k}")
    return optimal_k
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
    plt.show()

def cluster_clients(num_clients, feature_extractor=None, use_pca=True, pca_components=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix: Use same CNN architecture as FL training for consistency
    # Memory-efficient: Use smaller feature extractor instead of ResNet18
    if feature_extractor is None:
        from model import CNN
        model = CNN(num_classes=128)  # Use as feature extractor
        model.fc2 = nn.Identity()  # Remove final classification layer
    else:
        model = feature_extractor
    model.to(device)

    # Memory-efficient: Extract features in batches and clear cache
    client_features = []
    for cid in range(num_clients):
        feat = extract_features(cid, model, device)
        client_features.append(feat)
        
        # Clear GPU cache every 10 clients to prevent memory overflow
        if (cid + 1) % 10 == 0:
            torch.cuda.empty_cache()

    client_features = np.vstack(client_features)
    
    # Clear model from GPU memory after feature extraction
    model.cpu()
    torch.cuda.empty_cache()

    # 特徴量前処理（標準化 + 次元削減）
    processed_features = preprocess_features(client_features, use_pca=use_pca, n_components=pca_components)

    # クラスタ数決定
    optimal_k = determine_optimal_k(processed_features)

    # クラスタリング実行
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    cluster_ids = kmeans.fit_predict(processed_features)

    print(f"🔍 決定されたクラスタ数: {optimal_k}")
    visualize_clusters(processed_features, np.array([cluster_ids[cid] for cid in range(num_clients)]))

    return {cid: int(cluster_ids[cid]) for cid in range(num_clients)}
