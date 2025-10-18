import json
import os
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt

# ============================================================
# ================== パラメータ設定 =========================
# ============================================================
params = {
    'num_clients': 100,
    'metadata_weight': None,         # Noneで自動計算
    'method': 'distance',            # 'concat' / 'distance'
    'k_range': range(2, 7),
    'alpha_grid': [0.0, 0.25, 0.5, 0.75, 1.0],
    'use_mds_for_visual': True,
    'mds_dim': 2,
    'random_state': 42
}

# ============================================================
# ================== 乱数シード固定 =========================
# ============================================================
SEED = params['random_state']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# ================== 特徴抽出関数 =========================
# ============================================================
def extract_features(client_id, model, device):
    """クライアント単位で特徴を抽出"""
    dataset, _ = get_partitioned_data(client_id, params['num_clients'])
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
# ================== 可視化・評価関数 ======================
# ============================================================
def visualize_clusters(features, cluster_ids, title="Client Feature Clusters (t-SNE 2D Projection)"):
    tsne = TSNE(n_components=2, random_state=params['random_state'], perplexity=5)
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    for cluster in np.unique(cluster_ids):
        idx = cluster_ids == cluster
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f'Cluster {cluster}', alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
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
# ================== クラスタリング関数 ====================
# ============================================================
def cluster_clients_with_metadata_ratio(feature_extractor=None):
    """
    Args:
        feature_extractor: PyTorchモデル. Noneの場合デフォルトCNN
    Returns:
        dict: {client_id: cluster_id}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデル準備
    if feature_extractor is None:
        from model import CNN
        model = CNN(num_classes=38)
        model.fc2 = nn.Identity()
    else:
        model = feature_extractor
    model.to(device)

    # 画像特徴抽出
    client_features = []
    for cid in range(params['num_clients']):
        feat = extract_features(cid, model, device)
        client_features.append(feat)
    client_features = np.vstack(client_features)

    # クライアントごとのラベル比率ベクトル
    client_label_stats = []
    all_crops, all_diseases = set(), set()
    for cid in range(params['num_clients']):
        dataset, _ = get_partitioned_data(cid, params['num_clients'])
        class_names = dataset.classes
        label_counts = defaultdict(int)
        for _, label in dataset:
            label_counts[class_names[label]] += 1
        client_label_stats.append(label_counts)
        for cname in class_names:
            if "___" in cname:
                crop, disease = cname.split("___")
            else:
                crop, disease = cname, "Unknown"
            all_crops.add(crop)
            all_diseases.add(disease)

    crop_list = sorted(list(all_crops))
    disease_list = sorted(list(all_diseases))

    def compute_ratio_vector(stats):
        crop_counts = defaultdict(int)
        disease_counts = defaultdict(int)
        total = sum(stats.values()) if sum(stats.values()) > 0 else 1
        for cname, count in stats.items():
            if "___" in cname:
                crop, disease = cname.split("___")
            else:
                crop, disease = cname, "Unknown"
            crop_counts[crop] += count
            disease_counts[disease] += count
        crop_ratio = np.array([crop_counts[c] / total for c in crop_list])
        disease_ratio = np.array([disease_counts[d] / total for d in disease_list])
        return np.concatenate([crop_ratio, disease_ratio])

    metadata_ratios = np.vstack([compute_ratio_vector(s) for s in client_label_stats])

    # メタデータ重み自動計算
    img_var = np.var(client_features)
    meta_var = np.var(metadata_ratios)
    metadata_weight = params['metadata_weight']
    if metadata_weight is None:
        metadata_weight = float(np.sqrt(img_var / (meta_var + 1e-12)))
        metadata_weight = float(np.clip(metadata_weight, 1e-3, 1e3))
    print(f"[Auto] metadata_weight = {metadata_weight:.4f}")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # -------------------------
    # 横結合 + KMeans
    # -------------------------
    if params['method'] == "concat":
        combined = np.hstack([client_features, metadata_ratios * metadata_weight])
        X = StandardScaler().fit_transform(combined)
        best_k, best_labels, best_score = None, None, -np.inf
        for k in params['k_range']:
            kmeans = KMeans(n_clusters=k, random_state=params['random_state'], n_init=20).fit(X)
            labels = kmeans.labels_
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
            combined_score = sil + ch/1000.0 - db/10.0
            if combined_score > best_score:
                best_score = combined_score
                best_k = k
                best_labels = labels
        print(f"[Concat] best_k={best_k}, silhouette={sil:.4f}")
        visualize_clusters(X, best_labels, title=f"Concat Clusters k={best_k}")
        return {cid: int(best_labels[cid]) for cid in range(params['num_clients'])}

    # -------------------------
    # 距離融合（Distance Fusion）
    # -------------------------
    elif params['method'] == "distance":
        img_scaled = StandardScaler().fit_transform(client_features)
        meta_scaled = StandardScaler().fit_transform(metadata_ratios * metadata_weight)
        d_img = pairwise_distances(img_scaled, metric="euclidean")
        d_meta = pairwise_distances(meta_scaled, metric="euclidean")
        best_alpha, best_k, best_labels, best_sil = None, None, None, -np.inf

        for alpha in params['alpha_grid']:
            D = alpha * d_img + (1.0 - alpha) * d_meta
            sigma = np.std(D) if np.std(D) > 1e-8 else 1.0
            A = np.exp(-D / (sigma + 1e-12))
            for k in params['k_range']:
                sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=params['random_state'], n_init=10)
                labels = sc.fit_predict(A)
                try:
                    sil = silhouette_score(D, labels, metric="precomputed")
                except:
                    sil = -1.0
                if sil > best_sil:
                    best_sil = sil
                    best_alpha = alpha
                    best_k = k
                    best_labels = labels

        print(f"[DistanceFusion] best_alpha={best_alpha}, k={best_k}, silhouette={best_sil:.4f}")

        # 可視化
        if params['use_mds_for_visual']:
            try:
                mds = MDS(n_components=2, dissimilarity="precomputed", random_state=params['random_state'])
                X2 = mds.fit_transform(best_alpha * d_img + (1.0 - best_alpha) * d_meta)
                plt.figure(figsize=(8,6))
                for cl in np.unique(best_labels):
                    idx = best_labels == cl
                    plt.scatter(X2[idx,0], X2[idx,1], label=f"Cluster {cl}", alpha=0.7)
                plt.legend()
                plt.title(f"DistanceFusion alpha={best_alpha}, k={best_k}")
                plt.savefig(os.path.join(results_dir, "distancefusion_plot.png"), dpi=300, bbox_inches="tight")
                plt.show()
            except Exception as e:
                print("[Warning] MDS visualization failed:", e)

        return {cid: int(best_labels[cid]) for cid in range(params['num_clients'])}

    else:
        raise ValueError("method must be 'concat' or 'distance'")
