import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
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


def determine_optimal_k(features, k_range=(2, 10)):
    silhouettes = []

    for k in range(k_range[0], k_range[1] + 1):
        clustering = SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors', n_neighbors=10)
        cluster_ids = clustering.fit_predict(features)
        score = silhouette_score(features, cluster_ids)
        silhouettes.append(score)

    best_silhouette_k = np.argmax(silhouettes) + k_range[0]

    print(f"🧠 シルエット法による k: {best_silhouette_k}")
    return best_silhouette_k


def cluster_clients(num_clients, feature_extractor=None, use_pca=True, pca_components=50):
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
    processed_features = preprocess_features(client_features, use_pca=use_pca, n_components=pca_components)

    optimal_k = determine_optimal_k(processed_features)

    clustering = SpectralClustering(n_clusters=optimal_k, random_state=42, affinity='nearest_neighbors', n_neighbors=10)
    cluster_ids = clustering.fit_predict(processed_features)

    print(f"🔍 決定されたクラスタ数: {optimal_k}")
    return {cid: int(cluster_ids[cid]) for cid in range(num_clients)}
