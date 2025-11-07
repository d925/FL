# cluster_emb_patched.py
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

from utils import get_partitioned_data
from config import num_clients

# ============================================================
# ================== パラメータ設定 =========================
params = {
    'method': 'distance',
    'cluster_method': 'spectral',
    'k_range': range(3, 10),
    'alpha_grid': np.linspace(0.2, 0.8, 7).tolist(),
    'metadata_weight': 0.5,
    'use_mds_for_visual': True,
    'mds_dim': 2,
    'random_state': 42
}
params.update({
    'wasserstein_mode': 'sliced',  # '1d' or 'sliced'
    'L': 64,                       # sliced projection count
    'subsample': 800,              # subsample per client (None=use all)
    'pca_dim': 128,                 # reduce dimension for speed (None=keep)
    'n_jobs': 8                    # CPU parallel workers
})
# ============================================================

# 乱数シード固定
SEED = params['random_state']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------- ヘルパ関数：Sliced WD --------------------
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
    """
    Sliced Wasserstein distance matrix (possibly with PCA preproc).
    Returns: (dist_matrix, client_features_list_pca)
    """
    rng = np.random.RandomState(seed)

    # PCA 前処理（オプション） — 学習サンプルは各クライアントから代表を抽出
    if pca_dim is not None:
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

    # 投影ベクトル作成（ユニット長に正規化）
    Vs = rng.normal(size=(L, d)).astype(np.float32)
    Vs /= np.linalg.norm(Vs, axis=1, keepdims=True) + 1e-12

    # 各投影で投影値を算出してソート（並列）
    sorted_per_proj = [None] * L

    def job(k):
        r = np.random.RandomState(seed + k)
        return _proj_and_sort_for_projection(k, Vs[k], client_features_list, subsample=subsample, rng=r)

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_jobs, L)) as exe:
        futures = {exe.submit(job, k): k for k in range(L)}
        for fut in concurrent.futures.as_completed(futures):
            k = futures[fut]
            sorted_per_proj[k] = fut.result()
    proj_time = time.time() - start

    # 各投影についてペアワイズWDを計算して平均
    start = time.time()
    dist = np.zeros((n_clients, n_clients), dtype=np.float64)
    for k in range(L):
        dist += _sliced_wasserstein_pairwise(sorted_per_proj[k])
    dist /= float(L)
    wd_time = time.time() - start

    print(f"[SlicedW] projection_time={proj_time:.2f}s, pairwise_time={wd_time:.2f}s (L={L}, d={d})")
    return dist, client_features_list

# ============================================================
# ================== 特徴抽出関数 ============================
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
# ================== 可視化・評価関数 ============================
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
    Distribution を保持したまま Wasserstein（切片 / sliced）で距離行列を作成し、
    メタデータ距離と統計的に整合的に融合してスペクトラルクラスタリングする。
    """
    metadata_weight   = params.get('metadata_weight', None)
    method            = params.get('method', 'distance')
    cluster_method    = params.get('cluster_method', 'spectral')
    k_range           = params.get('k_range', range(2, 7))
    alpha_grid        = params.get('alpha_grid', [0.0, 0.25, 0.5, 0.75, 1.0])
    use_mds_for_visual= params.get('use_mds_for_visual', True)
    random_state      = params.get('random_state', 42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデル準備（特徴抽出器）
    if feature_extractor is None:
        from model import CNN
        model = CNN(num_classes=38)
        model.fc2 = nn.Identity()
    else:
        model = feature_extractor
    model.to(device)

    # --- 画像特徴抽出（各クライアント全サンプル） ---
    client_features_list_raw = []
    for cid in range(num_clients):
        feat = extract_features(cid, model, device)
        client_features_list_raw.append(feat)

    # --- Distribution branch: compute dist_img and ensure client_features (PCA-space) consistency ---
    if use_distribution:
        if params.get('wasserstein_mode', 'sliced') == 'sliced':
            dist_img, client_features_list = compute_sliced_wasserstein_matrix(
                client_features_list_raw,
                L=params.get('L', 64),
                subsample=params.get('subsample', None),
                pca_dim=params.get('pca_dim', None),
                n_jobs=params.get('n_jobs', 8),
                seed=params.get('random_state', 42)
            )
            # client_features_list は PCA 後の list[(n_i, pca_dim)]（もし pca_dim None なら raw のまま）
            client_features_mean = np.vstack([cf.mean(axis=0) for cf in client_features_list])
        else:
            # 1D per-dim Wasserstein (legacy) — PCA 統一を行う（pca_dim が指定されていれば）
            pca_dim = params.get('pca_dim', None)
            if pca_dim is not None:
                rng = np.random.RandomState(params.get('random_state', 42))
                samples = []
                for Xi in client_features_list_raw:
                    take = min(Xi.shape[0], 500)
                    idx = rng.choice(Xi.shape[0], size=take, replace=False)
                    samples.append(Xi[idx])
                concat = np.vstack(samples)
                pca = PCA(n_components=pca_dim, random_state=params.get('random_state', 42))
                pca.fit(concat)
                client_features_list = [pca.transform(X) for X in client_features_list_raw]
            else:
                client_features_list = client_features_list_raw

            n_clients_local = len(client_features_list)
            dist_img = np.zeros((n_clients_local, n_clients_local))
            start_time = time.time()
            for i in range(n_clients_local):
                for j in range(i + 1, n_clients_local):
                    wd = np.mean([
                        wasserstein_distance(client_features_list[i][:, d],
                                            client_features_list[j][:, d])
                        for d in range(client_features_list[i].shape[1])
                    ])
                    dist_img[i, j] = dist_img[j, i] = wd
            print(f"[1DWD] elapsed={time.time() - start_time:.2f}s")
            client_features_mean = np.vstack([cf.mean(axis=0) for cf in client_features_list])

    # --- メタデータ比率ベクトル作成（crop + disease を分割してベクトル化） ---
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
                crop, disease = cname.split("___", 1)
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
                crop, disease = cname.split("___", 1)
            else:
                crop, disease = cname, "Unknown"
            crop_counts[crop] += count
            disease_counts[disease] += count
        crop_ratio = np.array([crop_counts[c] / total for c in crop_list])
        disease_ratio = np.array([disease_counts[d] / total for d in disease_list])
        return np.concatenate([crop_ratio, disease_ratio])

    metadata_ratios = np.vstack([compute_ratio_vector(s) for s in client_label_stats])

    # ---------------- metadata weighting: Standardize -> weight -> scale ----------------
    # Standardize meta first
    meta_scaled = StandardScaler().fit_transform(metadata_ratios)
    # Standardize image-mean vectors (in PCA-space)
    img_scaled = StandardScaler().fit_transform(client_features_mean)

    # compute representative variances (mean of per-dim variances)
    img_var = float(np.mean(np.var(img_scaled, axis=0)))
    meta_var = float(np.mean(np.var(meta_scaled, axis=0)))
    metadata_weight = params.get('metadata_weight', None)
    if metadata_weight is None:
        metadata_weight = float(np.sqrt((img_var + 1e-12) / (meta_var + 1e-12)))
        metadata_weight = float(np.clip(metadata_weight, 1e-3, 1e3))
    # apply weight AFTER standardization
    meta_scaled = meta_scaled * metadata_weight
    print(f"[Auto] metadata_weight = {metadata_weight:.4f}")

    # ---------------- compute distance matrices ----------------
    dist_meta = pairwise_distances(meta_scaled, metric="euclidean")

    # normalize distance matrices by std to make them comparable (statistical justification)
    eps = 1e-12
    dist_img_std = np.std(dist_img) if np.std(dist_img) > 0 else 1.0
    dist_meta_std = np.std(dist_meta) if np.std(dist_meta) > 0 else 1.0
    dist_img_norm = dist_img / (dist_img_std + eps)
    dist_meta_norm = dist_meta / (dist_meta_std + eps)

    # ---------------- fusion and clustering (RBF affinity with proper formula) ----------------
    best_sil, best_alpha, best_k, best_labels = -1.0, None, None, None
    for alpha in alpha_grid:
        D_fused = alpha * dist_img_norm + (1.0 - alpha) * dist_meta_norm

        # formal RBF kernel: exp(-D^2 / (2 sigma^2))
        sigma_final = np.std(D_fused) if np.std(D_fused) > 1e-12 else 1.0
        affinity = np.exp(- (D_fused ** 2) / (2.0 * (sigma_final ** 2) + eps))

        # spectral clustering (affinity must be symmetric)
        for k in k_range:
            sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=random_state, n_init=10)
            labels = sc.fit_predict(affinity)
            # silhouette on precomputed distances
            try:
                sil = silhouette_score(D_fused, labels, metric='precomputed')
            except Exception:
                sil = -1.0
            if sil > best_sil:
                best_sil, best_alpha, best_k, best_labels = float(sil), float(alpha), int(k), labels.copy()
                print(f"[BestUpdate] best_alpha={best_alpha}, best_k={best_k}, silhouette={best_sil:.4f}")

    print(f"[Distribution+MetadataFusion] best_alpha={best_alpha}, best_k={best_k}, silhouette={best_sil:.4f}")

    # optional: visualize using MDS on best fused distance
    if use_mds_for_visual and best_labels is not None:
        try:
            D_best = best_alpha * dist_img_norm + (1.0 - best_alpha) * dist_meta_norm
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state)
            X2 = mds.fit_transform(D_best)
            plt.figure(figsize=(8, 6))
            for cl in np.unique(best_labels):
                idx = best_labels == cl
                plt.scatter(X2[idx, 0], X2[idx, 1], label=f'Cluster {cl}', alpha=0.7)
            plt.legend()
            plt.title(f"Distribution+Metadata Fusion alpha={best_alpha}, k={best_k}")
            os.makedirs("results", exist_ok=True)
            plt.savefig(os.path.join("results", "distancefusion_plot.png"), dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print("[Warning] MDS visualization failed:", e)

    # return mapping
    return {cid: int(best_labels[cid]) for cid in range(num_clients)}
