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
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import time
import concurrent.futures
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

from utils import get_partitioned_data
from config import num_clients

# ============================================================
# ================== パラメータ設定 =========================
params = {
    'method': 'distance',            
    'cluster_method': 'spectral',    
    'k_range': range(3, 10),         
    'alpha_grid': np.linspace(0.2, 0.8, 7).tolist(),  
    'metadata_weight': None,         
    'use_mds_for_visual': True,
    'mds_dim': 2,
    'random_state': 42
}
params.update({
    'wasserstein_mode': 'sliced',  # '1d' or 'sliced'
    'L': 64,                       # sliced projection count
    'subsample': 800,              # subsample per client (None=use all)
    'pca_dim': 64,                 # reduce dimension for speed (None=keep)
    'n_jobs': 8                    # CPU parallel workers
})

# ============================================================
# ================== 乱数シード固定 =========================
SEED = params['random_state']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def _proj_and_sort_for_projection(k, v, client_features_list, subsample=None, rng=None):
    sorted_vals = []
    for Xi in client_features_list:
        if subsample is not None and Xi.shape[0] > subsample:
            idx = rng.choice(Xi.shape[0], size=subsample, replace=False)
            Xi_use = Xi[idx]
        else:
            Xi_use = Xi
        proj = Xi_use.dot(v)
        proj.sort()
        sorted_vals.append(proj)
    return sorted_vals

def _sliced_wasserstein_pairwise(sorted_vals):
    n = len(sorted_vals)
    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        xi = sorted_vals[i]
        for j in range(i + 1, n):
            xj = sorted_vals[j]
            M[i, j] = M[j, i] = wasserstein_distance(xi, xj)
    return M

def compute_sliced_wasserstein_matrix(client_features_list,
                                      L=64,
                                      subsample=None,
                                      pca_dim=None,
                                      n_jobs=8,
                                      seed=42):
    rng = np.random.RandomState(seed)

    # PCA
    if pca_dim is not None:
        # sample small subset for PCA fit
        samples = []
        for Xi in client_features_list:
            take = min(Xi.shape[0], 500)
            idx = rng.choice(Xi.shape[0], size=take, replace=False)
            samples.append(Xi[idx])
        concat = np.vstack(samples)
        pca = PCA(n_components=pca_dim, random_state=seed)
        pca.fit(concat)
        client_features_list = [pca.transform(X) for X in client_features_list]

    n_clients = len(client_features_list)
    d = client_features_list[0].shape[1]

    # create random projections
    Vs = rng.normal(size=(L, d)).astype(np.float32)
    Vs /= np.linalg.norm(Vs, axis=1, keepdims=True) + 1e-12

    # sort projection values for all clients (parallel)
    sorted_per_proj = [None] * L

    def job(k):
        r = np.random.RandomState(seed + k)
        return _proj_and_sort_for_projection(k, Vs[k], client_features_list,
                                             subsample=subsample, rng=r)

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_jobs, L)) as exe:
        futures = {exe.submit(job, k): k for k in range(L)}
        for fut in concurrent.futures.as_completed(futures):
            k = futures[fut]
            sorted_per_proj[k] = fut.result()
    proj_time = time.time() - start

    # accumulate pairwise WD
    start = time.time()
    dist = np.zeros((n_clients, n_clients), dtype=np.float64)
    for k in range(L):
        dist += _sliced_wasserstein_pairwise(sorted_per_proj[k])
    dist /= float(L)
    wd_time = time.time() - start

    print(f"[SlicedW] projection_time={proj_time:.2f}s, pairwise_time={wd_time:.2f}s")
    return dist

# ============================================================
# ================== 特徴抽出関数 =========================
def extract_features(client_id, model, device):
    """クライアント単位で特徴を抽出（全サンプル保持）"""
    dataset, _ = get_partitioned_data(client_id, num_clients)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    features = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            feat = model(x)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)  # 平均はせず全サンプル保持

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

# ============================================================
# ================== クラスタリング関数 ====================
def cluster_clients_with_metadata_ratio(num_clients, feature_extractor=None, use_distribution=True):
    """
    params 辞書をグローバル参照してクラスタリングを行う。
    use_distribution=True の場合、各クライアントの特徴分布をそのまま使って
    Wasserstein距離で距離行列を作成してクラスタリング。
    """
    metadata_weight   = params.get('metadata_weight', None)
    method            = params.get('method', 'distance')
    cluster_method    = params.get('cluster_method', 'spectral')
    k_range           = params.get('k_range', range(2, 7))
    alpha_grid        = params.get('alpha_grid', [0.0, 0.25, 0.5, 0.75, 1.0])
    use_mds_for_visual= params.get('use_mds_for_visual', True)
    random_state      = params.get('random_state', 42)

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
    client_features_list = []
    for cid in range(num_clients):
        feat = extract_features(cid, model, device)
        client_features_list.append(feat)

    # クライアント平均ベクトル（既存手法用）
    client_features = np.vstack([cf.mean(axis=0) for cf in client_features_list])

    # メタデータ比率ベクトル
    client_label_stats = []
    all_crops, all_diseases = set(), set()
    for cid in range(num_clients):
        dataset, _ = get_partitioned_data(cid, num_clients)
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
    if metadata_weight is None:
        metadata_weight = float(np.sqrt(img_var / (meta_var + 1e-12)))
        metadata_weight = float(np.clip(metadata_weight, 1e-3, 1e3))
    print(f"[Auto] metadata_weight = {metadata_weight:.4f}")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 標準化
    img_scaled = StandardScaler().fit_transform(client_features)
    meta_scaled = StandardScaler().fit_transform(metadata_ratios * metadata_weight)

    # ==== 分布そのままの距離行列を作る手法 ====
    if use_distribution:
        wasser_mode = params.get('wasserstein_mode', '1d')

        if wasser_mode == '1d':
            print("[Distribution] Using original 1D Wasserstein")
            start_time = time.time()
            n_clients = len(client_features_list)
            dist_img = np.zeros((n_clients, n_clients))
            for i in range(n_clients):
                for j in range(i + 1, n_clients):
                    wd = np.mean([
                        wasserstein_distance(client_features_list[i][:, d],
                                            client_features_list[j][:, d])
                        for d in range(client_features_list[i].shape[1])
                    ])
                    dist_img[i, j] = dist_img[j, i] = wd
            print(f"[1DWD] elapsed={time.time() - start_time:.2f}s")

        elif wasser_mode == 'sliced':
            print("[Distribution] Using Sliced Wasserstein")
            start_time = time.time()
            dist_img = compute_sliced_wasserstein_matrix(
                client_features_list,
                L=params.get('L', 64),
                subsample=params.get('subsample', None),
                pca_dim=params.get('pca_dim', None),
                n_jobs=params.get('n_jobs', 8),
                seed=params.get('random_state', 42)
            )
            print(f"[SlicedWD] total_elapsed={time.time() - start_time:.2f}s")

        else:
            raise ValueError("wasserstein_mode must be '1d' or 'sliced'")

        print("[Distribution] Wasserstein距離行列を作成")
        # メタデータ距離行列
        dist_meta = pairwise_distances(metadata_ratios * metadata_weight, metric="euclidean")

        # αグリッド探索して最適クラスタリング
        best_sil, best_alpha, best_k, best_labels = -1, None, None, None
        for alpha in alpha_grid:
            D = alpha * dist_img + (1 - alpha) * dist_meta
            sigma = np.std(D) if np.std(D) > 1e-8 else 1.0
            affinity = np.exp(-D / (sigma + 1e-12))

            for k in k_range:
                sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                        random_state=random_state, n_init=10)
                labels = sc.fit_predict(affinity)
                try:
                    sil = silhouette_score(D, labels, metric="precomputed")
                except:
                    sil = -1
                if sil > best_sil:
                    best_sil, best_alpha, best_k, best_labels = sil, alpha, k, labels

        print(f"[Distribution+MetadataFusion] best_alpha={best_alpha}, best_k={best_k}, silhouette={best_sil:.4f}")

        return {cid: int(best_labels[cid]) for cid in range(n_clients)}

    # ==== 既存 distance / concat 手法 ====
    # （ここは元の distance / concat のコードを流用）
    # client_features = 平均ベクトル
    # ... (略, 既存の distance / concat 処理をそのまま使える)

    # ---- method = concat ----
    if method == "concat":
        X = np.hstack([img_scaled, meta_scaled])
        if cluster_method == "kmeans":
            best_k, best_labels, best_score = None, None, -np.inf
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20).fit(X)
                labels = kmeans.labels_
                sil = silhouette_score(X, labels)
                ch = calinski_harabasz_score(X, labels)
                db = davies_bouldin_score(X, labels)
                combined_score = sil + ch / 1000.0 - db / 10.0
                if combined_score > best_score:
                    best_score, best_k, best_labels = combined_score, k, labels
            print(f"[Concat][KMeans] best_k={best_k}, silhouette={sil:.4f}")

        elif cluster_method == "hierarchical":
            best_labels = AgglomerativeClustering(n_clusters=max(k_range), linkage='ward').fit_predict(X)
            print(f"[Concat][Hierarchical] clusters formed: {len(np.unique(best_labels))}")

        elif cluster_method == "dbscan":
            best_labels = DBSCAN(metric='euclidean', eps=0.5, min_samples=5).fit_predict(X)
            print(f"[Concat][DBSCAN] clusters formed: {len(np.unique(best_labels))}")

        else:
            raise ValueError("Unsupported cluster_method for concat")

        visualize_clusters(X, best_labels, title=f"Concat Clusters ({cluster_method})")
        return {cid: int(best_labels[cid]) for cid in range(num_clients)}

    # ---- method = distance ----
    elif method == "distance":
        d_img = pairwise_distances(img_scaled, metric="euclidean")
        d_meta = pairwise_distances(meta_scaled, metric="euclidean")
        best_alpha, best_k, best_labels, best_sil = None, None, None, -np.inf

        for alpha in alpha_grid:
            D = alpha * d_img + (1.0 - alpha) * d_meta

            if cluster_method == "spectral":
                sigma = np.std(D) if np.std(D) > 1e-8 else 1.0
                A = np.exp(-D / (sigma + 1e-12))
                for k in k_range:
                    sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                            random_state=random_state, n_init=10)
                    labels = sc.fit_predict(A)
                    try:
                        sil = silhouette_score(D, labels, metric="precomputed")
                    except:
                        sil = -1.0
                    if sil > best_sil:
                        best_sil, best_alpha, best_k, best_labels = sil, alpha, k, labels

            elif cluster_method == "hierarchical":
                for k in k_range:
                    labels = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average').fit_predict(D)
                    try:
                        sil = silhouette_score(D, labels, metric="precomputed")
                    except:
                        sil = -1.0
                    if sil > best_sil:
                        best_sil, best_alpha, best_k, best_labels = sil, alpha, k, labels

            elif cluster_method == "dbscan":
                labels = DBSCAN(metric='precomputed', eps=0.5, min_samples=5).fit_predict(D)
                try:
                    sil = silhouette_score(D, labels, metric="precomputed")
                except:
                    sil = -1.0
                if sil > best_sil:
                    best_sil, best_alpha, best_k, best_labels = sil, alpha, len(np.unique(labels)), labels

            else:
                raise ValueError("Unsupported cluster_method for distance")

        print(f"[DistanceFusion] best_alpha={best_alpha}, k={best_k}, silhouette={best_sil:.4f}")

        if use_mds_for_visual:
            try:
                mds = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state)
                X2 = mds.fit_transform(best_alpha * d_img + (1.0 - best_alpha) * d_meta)
                plt.figure(figsize=(8, 6))
                for cl in np.unique(best_labels):
                    idx = best_labels == cl
                    plt.scatter(X2[idx, 0], X2[idx, 1], label=f"Cluster {cl}", alpha=0.7)
                plt.legend()
                plt.title(f"DistanceFusion {cluster_method} alpha={best_alpha}, k={best_k}")
                plt.savefig(os.path.join("results", "distancefusion_plot.png"), dpi=300, bbox_inches="tight")
                plt.show()
            except Exception as e:
                print("[Warning] MDS visualization failed:", e)

        return {cid: int(best_labels[cid]) for cid in range(num_clients)}

    else:
        raise ValueError("method must be 'concat' or 'distance'")
