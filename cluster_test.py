import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from utils import get_partitioned_data
from config import num_clients
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# -------------------------
# 地域マッピング (PlantVillageの14作物 → アフリカ地域)
# -------------------------
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

# -------------------------
# 特徴抽出
# -------------------------
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

# -------------------------
# クライアントのメタデータ取得
# -------------------------
def get_client_metadata(client_id):
    dataset, _ = get_partitioned_data(client_id, num_clients)
    labels = [y for _, y in dataset]
    label_str = dataset.classes[labels[0]]  # ex. "Apple___Apple_scab"

    if "___" in label_str:
        crop, disease = label_str.split("___")
    else:
        print(f"[WARN] Unexpected label format: {label_str}")
        crop, disease = label_str, "Unknown"

    region = crop_region_map.get(crop, "Unknown")
    return crop, disease, region
# -------------------------
# 前処理
# -------------------------
def preprocess_features(features, use_pca=True, n_components=50):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    if use_pca:
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(scaled)
        return reduced
    return scaled

# -------------------------
# 可視化
# -------------------------
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
    sil_score = silhouette_score(features, cluster_ids) if len(np.unique(cluster_ids)) > 1 else -1
    ch_score = calinski_harabasz_score(features, cluster_ids) if len(np.unique(cluster_ids)) > 1 else -1
    db_score = davies_bouldin_score(features, cluster_ids) if len(np.unique(cluster_ids)) > 1 else -1
    print(f"🔍 {method_name} | Silhouette={sil_score:.4f}, CH={ch_score:.2f}, DB={db_score:.4f}")
    return sil_score, ch_score, db_score

# -------------------------
# KMeans + 複数指標
# -------------------------
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
    elbow_k = k_range[np.argmin(deltas)+1]
    labels = KMeans(n_clusters=elbow_k, random_state=42).fit_predict(features)
    sil, ch, db = silhouette_score(features, labels), calinski_harabasz_score(features, labels), davies_bouldin_score(features, labels)
    print(f"エルボー法 k={elbow_k} | Sil={sil:.4f}, CH={ch:.2f}, DB={db:.4f}")

    return elbow_k

def determine_k_internal(features, k_range=range(2, 11)):
    scores = {}
    best_k = None
    best_score = -np.inf

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)

        sil = silhouette_score(features, labels)
        ch = calinski_harabasz_score(features, labels)
        db = davies_bouldin_score(features, labels)
        combined = sil + ch/1000 - db/10

        scores[k] = {"Sil": sil, "CH": ch, "DB": db, "Combined": combined}

        if combined > best_score:
            best_score = combined
            best_k = k
    best_metrics = scores[best_k]
    print(f"🧠 内部指標による最適k: {best_k} | Sil={best_metrics['Sil']:.4f}, CH={best_metrics['CH']:.2f}, DB={best_metrics['DB']:.4f}, Combined={best_metrics['Combined']:.4f}")

    return best_k

# -------------------------
# 画像特徴のみのクラスタリング
# -------------------------
def cluster_clients_kmeans_dual(num_clients, feature_extractor=None, use_pca=True, pca_components=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if feature_extractor is None:
        from model import CNN
        model = CNN(num_classes=38)
        model.fc2 = nn.Identity()
    else:
        model = feature_extractor
    model.to(device)

    client_features = [extract_features(cid, model, device) for cid in range(num_clients)]
    client_features = np.vstack(client_features)
    model.cpu()
    torch.cuda.empty_cache()

    processed_features = preprocess_features(client_features, use_pca=use_pca, n_components=pca_components)

    k_elbow = determine_k_elbow(processed_features)
    k_internal = determine_k_internal(processed_features)

    labels_elbow = KMeans(n_clusters=k_elbow, random_state=42).fit_predict(processed_features)
    labels_internal = KMeans(n_clusters=k_internal, random_state=42).fit_predict(processed_features)

    print("----- クラスタリング結果 (特徴量のみ) -----")
    visualize_clusters(processed_features, labels_elbow, title=f"Elbow KMeans Clusters (k={k_elbow})")
    visualize_clusters(processed_features, labels_internal, title=f"Internal KMeans Clusters (k={k_internal})")

    return {
        "elbow": {cid: int(labels_elbow[cid]) for cid in range(num_clients)},
        "internal": {cid: int(labels_internal[cid]) for cid in range(num_clients)}
    }

# -------------------------
# メタデータ込みのクラスタリング
# -------------------------
def cluster_clients_with_metadata(num_clients, feature_extractor=None, use_pca=True, pca_components=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if feature_extractor is None:
        from model import CNN
        model = CNN(num_classes=38)
        model.fc2 = nn.Identity()
    else:
        model = feature_extractor
    model.to(device)

    # 画像特徴抽出
    client_features = [extract_features(cid, model, device) for cid in range(num_clients)]
    client_features = np.vstack(client_features)

    # メタデータ抽出
    metadata = [get_client_metadata(cid) for cid in range(num_clients)]
    encoder = OneHotEncoder(sparse_output=False)
    metadata_encoded = encoder.fit_transform(metadata).astype(np.float32)
    combined_features = np.hstack([client_features, metadata_encoded])

    # 結合
    combined_features = np.hstack([client_features, metadata_encoded])

    model.cpu()
    torch.cuda.empty_cache()

    processed_features = preprocess_features(combined_features, use_pca=use_pca, n_components=pca_components)

    # エルボー法で最適kを決定
    k_elbow = determine_k_elbow(processed_features)

    # KMeansクラスタリング
    labels_elbow = KMeans(n_clusters=k_elbow, random_state=42).fit_predict(processed_features)

    print("----- クラスタリング結果 (メタデータ込み, エルボー法) -----")
    visualize_clusters(processed_features, labels_elbow, title=f"Metadata+Elbow KMeans Clusters (k={k_elbow})")

    # 返り値の形を統一（elbowだけ）
    return {cid: int(labels_elbow[cid]) for cid in range(num_clients)}
