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

def determine_optimal_k(features, k_range=(2, 10)):
    wcss = []  # クラスタ内誤差平方和（Within-Cluster Sum of Squares）
    silhouettes = []  # シルエット係数

    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_ids = kmeans.fit_predict(features)
        wcss.append(kmeans.inertia_)  # inertia_ = WCSS
        score = silhouette_score(features, cluster_ids)
        silhouettes.append(score)

    # エルボー法: WCSSの減少が鈍化するポイント（最も差分が小さくなるところ）
    deltas = np.diff(wcss)
    elbow_k = np.argmin(np.abs(np.diff(deltas))) + 2  # +2 because index starts from k=2

    # シルエット係数最大のk
    best_silhouette_k = np.argmax(silhouettes) + k_range[0]

    # 両方の平均値 or どちらか保守的な方を選ぶ
    optimal_k = max(elbow_k, best_silhouette_k)
    return optimal_k

def cluster_clients(num_clients, feature_extractor=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if feature_extractor is None:
        model = resnet18(pretrained=True)
        model.fc = nn.Identity()  # 全結合層除去
    else:
        model = feature_extractor
    model.to(device)

    client_features = []
    for cid in range(num_clients):
        feat = extract_features(cid, model, device)
        client_features.append(feat)

    client_features = np.vstack(client_features)
    optimal_k = determine_optimal_k(client_features)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    cluster_ids = kmeans.fit_predict(client_features)

    print(f"🔍 決定されたクラスタ数: {optimal_k}")
    return {cid: int(cluster_ids[cid]) for cid in range(num_clients)}
