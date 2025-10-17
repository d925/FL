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
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS
from collections import defaultdict

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
# ============================================================
# 比率ベクトルを用いたメタデータ + 画像特徴クラスタリング
# ============================================================
def cluster_clients_with_metadata_ratio(
    num_clients,
    feature_extractor=None,
    metadata_weight=None,
    method="concat",            # "concat" または "distance"
    k_range=range(2, 7),        # 探索するクラスタ数範囲
    alpha_grid=None,            # 距離融合のとき探索する alpha 値リスト (画像重み)
    random_state=42
):
    """
    Args:
      method: "concat" -> 特徴横結合 + KMeans (従来)
              "distance" -> 距離融合: D = alpha * d_img + (1-alpha) * d_meta
      alpha_grid: list of alpha to try for distance fusion. If None, defaults to [0.0,0.25,0.5,0.75,1.0]
      k_range: cluster k を探索する範囲
    Returns:
      dict: {cid: cluster_id, ...} かつ results/cluster_metrics.json に内部指標等を保存
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature_extractor is None:
        from model import CNN
        model = CNN(num_classes=38)
        model.fc2 = nn.Identity()
    else:
        model = feature_extractor
    model.to(device)

    # 画像特徴を抽出
    client_features = []
    for cid in range(num_clients):
        feat = extract_features(cid, model, device)
        client_features.append(feat)
    client_features = np.vstack(client_features)  # shape (N_clients, feat_dim)

    # クライアントごとのラベル比率ベクトル (crop_ratio + disease_ratio)
    all_crops, all_diseases = set(), set()
    client_label_stats = []
    for cid in range(num_clients):
        dataset, _ = get_partitioned_data(cid, num_clients)
        class_names = dataset.classes
        label_counts = defaultdict(int)
        for _, label in dataset:
            label_counts[class_names[label]] += 1
        client_label_stats.append(label_counts)
        # collect all crop/disease types across dataset.classes
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

    metadata_ratios = np.vstack([compute_ratio_vector(s) for s in client_label_stats])  # shape (N_clients, C+D)

    # 自動メタデータ重み（concat時のスケーリング）
    img_var = np.var(client_features)
    meta_var = np.var(metadata_ratios)
    if metadata_weight is None:
        metadata_weight = float(np.sqrt(img_var / (meta_var + 1e-12)))
    # ただし大きすぎる値はクリップ（数値安定化のため）
    metadata_weight = float(np.clip(metadata_weight, 1e-3, 1e3))
    print(f"[Auto] metadata_weight = {metadata_weight:.4f}")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    metrics_out = {
        "method": method,
        "metadata_weight": metadata_weight,
        "num_clients": int(num_clients),
        "crop_list": crop_list,
        "disease_list": disease_list,
    }

    # -------------------------
    # 方法A: 横結合 + KMeans
    # -------------------------
    if method == "concat":
        combined = np.hstack([client_features, metadata_ratios * metadata_weight])
        scaler = StandardScaler()
        X = scaler.fit_transform(combined)

        # k選択：内部指標のcombinedスコアを最大化
        best_k, best_score = None, -np.inf
        best_labels = None
        scores_by_k = {}
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20).fit(X)
            labels = kmeans.labels_
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
            combined_score = sil + ch / 1000.0 - db / 10.0
            scores_by_k[int(k)] = {"silhouette": float(sil), "calinski_harabasz": float(ch), "davies_bouldin": float(db), "combined": float(combined_score)}
            if combined_score > best_score:
                best_score = combined_score
                best_k = int(k)
                best_labels = labels
        metrics_out.update({"k_candidates": list(k_range), "k_scores": scores_by_k, "k_opt": best_k})
        sil, ch, db = scores_by_k[best_k]["silhouette"], scores_by_k[best_k]["calinski_harabasz"], scores_by_k[best_k]["davies_bouldin"]

        # 保存
        metrics_out.update({"silhouette": float(sil), "calinski_harabasz": float(ch), "davies_bouldin": float(db)})
        with open(os.path.join(results_dir, "cluster_metrics.json"), "w") as f:
            json.dump(metrics_out, f, indent=2)

        # 可視化
        visualize_clusters(X, best_labels, title=f"Concat Image+MetaRatio KMeans (k={best_k})")
        return {cid: int(best_labels[cid]) for cid in range(num_clients)}

    # -------------------------
    # 方法B: 距離融合（distance fusion）
    # -------------------------
    elif method == "distance":
        # スケールしてから距離計算（それぞれの空間で正規化）
        img_scaled = StandardScaler().fit_transform(client_features)
        meta_scaled = StandardScaler().fit_transform(metadata_ratios * metadata_weight)

        # 距離行列
        d_img = pairwise_distances(img_scaled, metric="euclidean")
        d_meta = pairwise_distances(meta_scaled, metric="euclidean")

        # alpha のグリッド探索（画像の重み alpha）
        if alpha_grid is None:
            alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

        best_alpha, best_k, best_labels = None, None, None
        best_sil = -np.inf
        k_scores_for_alpha = {}

        for alpha in alpha_grid:
            D = alpha * d_img + (1.0 - alpha) * d_meta  # 融合距離
            # 距離 -> 類似度 (RBF like): A = exp(-D / sigma)
            sigma = np.std(D) if np.std(D) > 1e-8 else 1.0
            A = np.exp(-D / (sigma + 1e-12))

            # k探索：SpectralClustering（affinity=precomputed）
            scores_k = {}
            for k in k_range:
                sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=random_state, n_init=10)
                labels = sc.fit_predict(A)
                # silhouette using precomputed distances
                try:
                    sil = silhouette_score(D, labels, metric="precomputed")
                except Exception:
                    sil = -1.0
                # CH/DB require Euclidean features; approximate by MDS embedding from D
                try:
                    mds = MDS(n_components=min(10, max(2, img_scaled.shape[1]//2)), dissimilarity="precomputed", random_state=random_state)
                    X_mds = mds.fit_transform(D)
                    ch = calinski_harabasz_score(X_mds, labels)
                    db = davies_bouldin_score(X_mds, labels)
                except Exception:
                    ch, db = -1.0, -1.0

                combined_score = sil + (ch / 1000.0 if ch > 0 else 0.0) - (db / 10.0 if db > 0 else 0.0)
                scores_k[int(k)] = {"silhouette": float(sil), "calinski_harabasz": float(ch), "davies_bouldin": float(db), "combined": float(combined_score)}
                # pick best for this alpha
                if sil > best_sil:
                    best_sil = sil
                    best_alpha = float(alpha)
                    best_k = int(k)
                    best_labels = labels
            k_scores_for_alpha[float(alpha)] = scores_k

        # まとめ保存
        metrics_out.update({
            "alpha_grid": alpha_grid,
            "best_alpha": best_alpha,
            "k_candidates": list(k_range),
            "k_scores_by_alpha": k_scores_for_alpha,
            "k_opt": int(best_k) if best_k is not None else None,
            "silhouette": float(best_sil)
        })
        with open(os.path.join(results_dir, "cluster_metrics.json"), "w") as f:
            json.dump(metrics_out, f, indent=2)

        print(f"[DistanceFusion] best_alpha={best_alpha}, k_opt={best_k}, silhouette={best_sil:.4f}")
        # 可視化用に MDS 埋め込みして投影（Spectral のラベルで）
        try:
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state)
            X2 = mds.fit_transform(best_alpha * d_img + (1.0 - best_alpha) * d_meta)
            plt.figure(figsize=(8,6))
            for cl in np.unique(best_labels):
                idx = best_labels == cl
                plt.scatter(X2[idx,0], X2[idx,1], label=f"Cluster {cl}", alpha=0.7)
            plt.legend()
            plt.title(f"DistanceFusion (alpha={best_alpha}, k={best_k})")
            plt.savefig(os.path.join(results_dir, "distancefusion_plot.png"), dpi=300, bbox_inches="tight")
            plt.show()
        except Exception as e:
            print("[Warning] MDS visualization failed:", e)

        return {cid: int(best_labels[cid]) for cid in range(num_clients)}

    else:
        raise ValueError("method must be 'concat' or 'distance'")
