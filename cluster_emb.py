import json
import os
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
# クライアントメタデータ取得
# ============================================================
def get_client_metadata(client_id):
    dataset, _ = get_partitioned_data(client_id, num_clients)
    labels = [y for _, y in dataset]
    label_str = dataset.classes[labels[0]]

    if "___" in label_str:
        crop, disease = label_str.split("___")
    else:
        crop, disease = label_str, "Unknown"

    region = crop_region_map.get(crop, "Unknown")
    return crop, disease, region

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
def cluster_clients_with_metadata_emb(num_clients, feature_extractor=None, metadata_weight=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature_extractor is None:
        from model import CNN
        model = CNN(num_classes=38)
        model.fc2 = nn.Identity()
    else:
        model = feature_extractor
    model.to(device)

    # ---- 画像特徴抽出 ----
    client_features = [extract_features(cid, model, device) for cid in range(num_clients)]
    client_features = np.vstack(client_features)

    # ---- メタデータ抽出 ----
    metadata = [get_client_metadata(cid) for cid in range(num_clients)]
    crops, diseases, regions = zip(*metadata)
    crop_le, disease_le, region_le = LabelEncoder(), LabelEncoder(), LabelEncoder()
    crop_ids = torch.tensor(crop_le.fit_transform(crops), dtype=torch.long)
    disease_ids = torch.tensor(disease_le.fit_transform(diseases), dtype=torch.long)
    region_ids = torch.tensor(region_le.fit_transform(regions), dtype=torch.long)

    emb_model = MetadataEmbedding(
        num_crops=len(crop_le.classes_),
        num_diseases=len(disease_le.classes_),
        num_regions=len(region_le.classes_),
        emb_dim=32
    ).to(device)

    with torch.no_grad():
        metadata_emb = emb_model(crop_ids.to(device), disease_ids.to(device), region_ids.to(device)).cpu().numpy()

    # ---- 自動重み調整 ----
    # 特徴ごとの分散比に応じて metadata_weight を自動算出
    img_var = np.var(client_features)
    meta_var = np.var(metadata_emb)
    if metadata_weight is None:
        metadata_weight = np.sqrt(img_var / (meta_var + 1e-8))
        print(f"[Auto] metadata_weight = {metadata_weight:.3f}")

    combined_features = np.hstack([client_features, metadata_emb * metadata_weight])

    # ---- 二段階正規化 ----
    scaler1 = StandardScaler()
    inter_scaled = scaler1.fit_transform(combined_features)
    scaler2 = StandardScaler(with_std=False)
    processed_features = scaler2.fit_transform(inter_scaled)

    # ---- クラスタリング ----
    k_opt = determine_k_internal(processed_features)
    labels = KMeans(n_clusters=k_opt, random_state=42, n_init=20).fit_predict(processed_features)
    labels = KMeans(n_clusters=k_opt, random_state=42, n_init=20).fit_predict(processed_features)
    sil, ch, db = evaluate_clusters(processed_features, labels, "Metadata+Image Clustering")

    # ---- 追加：内部指標を保存 ----
    os.makedirs("results", exist_ok=True)
    metrics_path = os.path.join("results", "cluster_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "k_opt": int(k_opt)
        }, f, indent=2)
    print(f"📁 クラスタリング内部指標を {metrics_path} に保存しました。")

    visualize_clusters(processed_features, labels, title=f"MetadataEmb KMeans Clusters (k={k_opt})")
    return {cid: int(labels[cid]) for cid in range(num_clients)}