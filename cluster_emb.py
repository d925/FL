import json
import os
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from utils import get_partitioned_data
from config import num_clients
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ============================================================
# 乱数シード固定
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# 地域マッピング
# ============================================================
crop_region_map = {
    "Apple": "North Africa",
    "Blueberry": "South Africa",
    "Cherry": "North Africa",
    "Corn": "East Africa",
    "Grape": "North Africa",
    "Orange": "North Africa",
    "Peach": "South Africa",
    "Pepper": "West Africa",
    "Potato": "East Africa",
    "Raspberry": "South Africa",
    "Soybean": "East Africa",
    "Squash": "West Africa",
    "Strawberry": "South Africa",
    "Tomato": "West Africa",
}

# ============================================================
# 特徴抽出
# ============================================================
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

# ============================================================
# 特徴前処理（PCA除去版）
# ============================================================
def preprocess_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# ============================================================
# 可視化・評価
# ============================================================
def visualize_clusters(features, cluster_ids, title="Client Feature Clusters (t-SNE 2D Projection)"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    for cluster in np.unique(cluster_ids):
        idx = cluster_ids == cluster
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f'Cluster {cluster}', alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.xlabel("TSNE Dim 1")
    plt.ylabel("TSNE Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cluster_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_clusters(features, cluster_ids, method_name="Clustering"):
    if len(np.unique(cluster_ids)) > 1:
        sil_score = silhouette_score(features, cluster_ids)
        ch_score = calinski_harabasz_score(features, cluster_ids)
        db_score = davies_bouldin_score(features, cluster_ids)
    else:
        sil_score, ch_score, db_score = -1, -1, -1

    print(f"🔍 {method_name} | Silhouette={sil_score:.4f}, CH={ch_score:.2f}, DB={db_score:.4f}")
    return sil_score, ch_score, db_score

# ============================================================
# k最適化（Elbow法 + 内部指標）
# ============================================================
def determine_k_elbow(features, k_range=range(2, 11)):
    wss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        wss.append(kmeans.inertia_)

    plt.figure()
    plt.plot(k_range, wss, 'o-', color='blue')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WSS (inertia)")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.show()

    deltas = np.diff(wss)
    elbow_k = k_range[np.argmin(deltas) + 1]
    labels = KMeans(n_clusters=elbow_k, random_state=42).fit_predict(features)
    sil, ch, db = silhouette_score(features, labels), calinski_harabasz_score(features, labels), davies_bouldin_score(features, labels)
    print(f"エルボー法 k={elbow_k} | Sil={sil:.4f}, CH={ch:.2f}, DB={db:.4f}")

    return elbow_k

def determine_k_internal(features, k_range=range(2, 11)):
    best_k, best_score = None, -np.inf
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        sil = silhouette_score(features, labels)
        ch = calinski_harabasz_score(features, labels)
        db = davies_bouldin_score(features, labels)
        combined = sil + ch/1000 - db/10
        if combined > best_score:
            best_score = combined
            best_k = k
    print(f"🧠 内部指標による最適k: {best_k}")
    return best_k

# ============================================================
# メタデータ埋め込みクラス
# ============================================================
class MetadataEmbedding(nn.Module):
    def __init__(self, num_crops, num_diseases, num_regions, emb_dim=32):
        super().__init__()
        self.crop_emb = nn.Embedding(num_crops, emb_dim)
        self.disease_emb = nn.Embedding(num_diseases, emb_dim)
        self.region_emb = nn.Embedding(num_regions, emb_dim)

        # Xavier初期化（trainableにするためfreeze削除）
        for emb in [self.crop_emb, self.disease_emb, self.region_emb]:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, crop_ids, disease_ids, region_ids):
        crop_vec = self.crop_emb(crop_ids)
        disease_vec = self.disease_emb(disease_ids)
        region_vec = self.region_emb(region_ids)
        return torch.cat([crop_vec, disease_vec, region_vec], dim=1)

# ============================================================
# 画像特徴 + メタデータ埋め込みクラスタリング
# ============================================================
def get_client_metadata_distribution(client_id, num_clients, crop_region_map):
    dataset, _ = get_partitioned_data(client_id, num_clients)
    labels = [y for _, y in dataset]
    class_names = dataset.classes

    crops, diseases = [], []
    for label in labels:
        label_str = class_names[label]
        if "___" in label_str:
            crop, disease = label_str.split("___")
        else:
            crop, disease = label_str, "Unknown"
        crops.append(crop)
        diseases.append(disease)

    # 頻度分布を確率化
    def normalize(counter):
        total = sum(counter.values())
        return {k: v / total for k, v in counter.items()}

    crop_dist = normalize(Counter(crops))
    disease_dist = normalize(Counter(diseases))
    region_dist = normalize(Counter([crop_region_map.get(c, "Unknown") for c in crops]))

    return crop_dist, disease_dist, region_dist


# ============================================================
# メタデータ分布＋画像特徴のマルチモーダルクラスタリング
# ============================================================
def cluster_clients_with_metadata_emb(model, device, num_clients, crop_region_map):
    print("=== Clustering clients using image features + metadata distributions ===")

    # ---- クライアントごとの分布取得 ----
    metadata = [
        get_client_metadata_distribution(cid, num_clients, crop_region_map)
        for cid in range(num_clients)
    ]

    # 全クライアントで出現したcrop/disease/regionをキー集合として抽出
    all_crops = sorted(list({c for dist, _, _ in metadata for c in dist.keys()}))
    all_diseases = sorted(list({d for _, dist, _ in metadata for d in dist.keys()}))
    all_regions = sorted(list({r for _, _, dist in metadata for r in dist.keys()}))

    # ---- 各分布を固定順でベクトル化 ----
    def vectorize(dist, keys):
        return np.array([dist.get(k, 0.0) for k in keys])

    crop_vecs = np.vstack([vectorize(c, all_crops) for c, _, _ in metadata])
    disease_vecs = np.vstack([vectorize(d, all_diseases) for _, d, _ in metadata])
    region_vecs = np.vstack([vectorize(r, all_regions) for _, _, r in metadata])

    metadata_vecs = np.hstack([crop_vecs, disease_vecs, region_vecs])

    # ---- 画像特徴を抽出 ----
    client_features = np.vstack([
        extract_features(cid, model, device) for cid in range(num_clients)
    ])

    # ---- スケール調整（画像とメタデータをバランスさせる） ----
    img_var = np.var(client_features)
    meta_var = np.var(metadata_vecs)
    metadata_weight = np.sqrt(img_var / (meta_var + 1e-8))
    combined_features = np.hstack([client_features, metadata_vecs * metadata_weight])

    # ---- 正規化とクラスタリング ----
    scaler = StandardScaler()
    processed_features = scaler.fit_transform(combined_features)

    k_opt = determine_k_internal(processed_features)
    kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=20)
    labels = kmeans.fit_predict(processed_features)

    # ---- 内部指標を計算 ----
    sil, ch, db = evaluate_clusters(processed_features, labels, "Image+MetadataDist Clustering")

    # ---- 結果出力 ----
    print(f"[Cluster Summary]  k={k_opt}")
    print(f"  Silhouette: {sil:.4f}")
    print(f"  Calinski-Harabasz: {ch:.4f}")
    print(f"  Davies-Bouldin: {db:.4f}")

    return labels, k_opt, {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}