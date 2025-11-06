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
    'metadata_weight': None,
    'use_mds_for_visual': True,
    'mds_dim': 2,
    'random_state': 42
}
# ユーザ指定：PCA後の次元 256（あなたの選択）
params.update({
    'wasserstein_mode': 'sliced',
    'initial_L': 64,
    'refine_L': 192,
    'refine_topk': 6,
    'subsample': None,
    'pca_dim': 128,   # ← ユーザ指定
    'n_jobs': 8,
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
def _proj_and_sort_for_projection_local(v, client_features_list, subsample=None, rng=None):
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

def _sliced_wasserstein_pairwise_from_proj_local(sorted_vals):
    n = len(sorted_vals)
    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        xi = sorted_vals[i]
        for j in range(i+1, n):
            xj = sorted_vals[j]
            M[i, j] = M[j, i] = wasserstein_distance(xi, xj)
    return M

def compute_sliced_wasserstein_matrix(client_features_list,
                                      initial_L=64,
                                      refine_L=192,
                                      refine_topk=6,
                                      subsample=None,
                                      pca_dim=256,
                                      n_jobs=8,
                                      seed=42):
    """
    二段階Sliced Wasserstein距離行列（PCA適用済み空間で計算）を返す。
    coarse → refine の二段階で精密化。
    """
    rng = np.random.RandomState(seed)

    # PCA 前処理（オプション）。ここで client_features_list を PCA 後の空間に統一する。
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

    # coarse 投影ベクトル
    Vs_coarse = rng.normal(size=(initial_L, d)).astype(np.float32)
    Vs_coarse /= np.linalg.norm(Vs_coarse, axis=1, keepdims=True) + 1e-12

    # coarse 投影 -> 各投影ごとのソート済み投影値を並列計算
    start_proj = time.time()
    sorted_per_proj_coarse = [None] * initial_L
    def job_coarse(k):
        r = np.random.RandomState(seed + 1000 + k)
        return _proj_and_sort_for_projection_local(Vs_coarse[k], client_features_list, subsample=subsample, rng=r)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_jobs, initial_L)) as exe:
        futures = {exe.submit(job_coarse, k): k for k in range(initial_L)}
        for fut in concurrent.futures.as_completed(futures):
            k = futures[fut]
            sorted_per_proj_coarse[k] = fut.result()
    proj_coarse_time = time.time() - start_proj

    # coarse ペア距離平均化
    start_pair_coarse = time.time()
    dist_coarse = np.zeros((n_clients, n_clients), dtype=np.float64)
    for k in range(initial_L):
        dist_coarse += _sliced_wasserstein_pairwise_from_proj_local(sorted_per_proj_coarse[k])
    dist_coarse /= float(initial_L)
    pair_coarse_time = time.time() - start_pair_coarse

    # candidate ペア選出（各クライアントについて近傍 topk）
    topk = max(1, int(refine_topk))
    candidate_pairs = set()
    for i in range(n_clients):
        dists = dist_coarse[i].copy()
        dists[i] = np.inf
        idxs = np.argpartition(dists, topk)[:topk]
        for j in idxs:
            a, b = (i, j) if i < j else (j, i)
            candidate_pairs.add((a, b))
    candidate_pairs = sorted(candidate_pairs)
    n_candidates = len(candidate_pairs)

    # refinement ベクトル
    Vs_refine = rng.normal(size=(refine_L, d)).astype(np.float32)
    Vs_refine /= np.linalg.norm(Vs_refine, axis=1, keepdims=True) + 1e-12

    # refinement の投影ごとのソート値を並列計算
    start_proj_ref = time.time()
    sorted_per_proj_ref = [None] * refine_L
    def job_ref(k):
        r = np.random.RandomState(seed + 2000 + k)
        return _proj_and_sort_for_projection_local(Vs_refine[k], client_features_list, subsample=subsample, rng=r)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_jobs, refine_L)) as exe:
        futures = {exe.submit(job_ref, k): k for k in range(refine_L)}
        for fut in concurrent.futures.as_completed(futures):
            k = futures[fut]
            sorted_per_proj_ref[k] = fut.result()
    proj_ref_time = time.time() - start_proj_ref

    # 各 candidate ペアに対して refine_L 投影の平均WD を計算
    start_pair_ref = time.time()
    refined_wd = {}
    for a, b in candidate_pairs:
        acc = 0.0
        for k in range(refine_L):
            xi = sorted_per_proj_ref[k][a]
            xj = sorted_per_proj_ref[k][b]
            acc += wasserstein_distance(xi, xj)
        refined_wd[(a, b)] = (acc / float(refine_L))
    pair_ref_time = time.time() - start_pair_ref

    # coarse と refined をブレンド（ここでは単純平均）
    final_dist = dist_coarse.copy()
    for (a, b), w in refined_wd.items():
        final_dist[a, b] = final_dist[b, a] = (final_dist[a, b] + w) / 2.0

    total_time = proj_coarse_time + pair_coarse_time + proj_ref_time + pair_ref_time
    print(f"[SlicedW-multiscale] proj_coarse={proj_coarse_time:.2f}s, pair_coarse={pair_coarse_time:.2f}s, proj_ref={proj_ref_time:.2f}s, pair_ref={pair_ref_time:.2f}s, candidates={n_candidates}, total={total_time:.2f}s")

    return final_dist, client_features_list  # PCA後の client_features_list も返す

# ============================================================
# 特徴抽出関数（全サンプル保持）
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
    return np.concatenate(features, axis=0)

# ============================================================
# 可視化
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
# メイン：クラスタリング関数
def cluster_clients_with_metadata_ratio(num_clients, feature_extractor=None, use_distribution=True):
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

    # 画像特徴抽出（全サンプル）
    client_features_list_raw = []
    for cid in range(num_clients):
        feat = extract_features(cid, model, device)  # shape (n_i, D_raw)
        client_features_list_raw.append(feat)

    # Distribution branch: compute SWD (sliced or 1d)
    if use_distribution:
        if params.get('wasserstein_mode', 'sliced') == 'sliced':
            dist_img, client_features_list = compute_sliced_wasserstein_matrix(
                client_features_list_raw,
                initial_L=params.get('initial_L', 64),
                refine_L=params.get('refine_L', 192),
                refine_topk=params.get('refine_topk', 6),
                subsample=params.get('subsample', None),
                pca_dim=params.get('pca_dim', 256),  # PCA次元（指定どおり）
                n_jobs=params.get('n_jobs', 8),
                seed=params.get('random_state', 42)
            )
            # client_features_list は PCA 後の list[(n_i, pca_dim)]
            # 平均特徴ベクトルも PCA後の平均に統一（重要）
            client_features = np.vstack([cf.mean(axis=0) for cf in client_features_list])
        else:
            # 1D per-dim Wasserstein (legacy)。ここでも PCA をかけて次元を揃えるため PCA を適用。
            pca_dim = params.get('pca_dim', None)
            if pca_dim is not None:
                # fit PCA on representative samples
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

            n_clients = len(client_features_list)
            dist_img = np.zeros((n_clients, n_clients))
            start_time = time.time()
            for i in range(n_clients):
                for j in range(i + 1, n_clients):
                    wd = np.mean([
                        wasserstein_distance(client_features_list[i][:, d],
                                            client_features_list[j][:, d])
                        for d in range(client_features_list[i].shape[1])
                    ])
                    dist_img[i, j] = dist_img[j, i] = wd
            print(f"[1DWD] elapsed={time.time() - start_time:.2f}s")
            client_features = np.vstack([cf.mean(axis=0) for cf in client_features_list])

    # ---------------- Meta-data vector (crop + disease) ----------------
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

    # ---------------- metadata weighting: Standardize -> weight -> distance ----------------
    # Correct order: standardize first, then apply metadata_weight (scale)
    meta_scaled = StandardScaler().fit_transform(metadata_ratios)  # standardized
    # auto compute metadata_weight if None (based on PCA-mean features variance)
    img_var = np.var(client_features)
    meta_var = np.var(meta_scaled)
    metadata_weight = params.get('metadata_weight', None)
    if metadata_weight is None:
        metadata_weight = float(np.sqrt((img_var + 1e-12) / (meta_var + 1e-12)))
        metadata_weight = float(np.clip(metadata_weight, 1e-3, 1e3))
    # apply weight AFTER standardization
    meta_scaled = meta_scaled * metadata_weight
    print(f"[Auto] metadata_weight = {metadata_weight:.4f}")

    # ---------------- compute distance matrices ----------------
    # client_features are PCA-means (already computed)
    # standardize image-mean vectors BEFORE any concatenation/distance to keep consistent scaling
    img_scaled = StandardScaler().fit_transform(client_features)

    # dist_img already computed for distribution branch (Sliced/1D)
    # dist_meta computed from meta_scaled
    dist_meta = pairwise_distances(meta_scaled, metric="euclidean")

    # ---------------- fusion and clustering ----------------
    best_sil, best_alpha, best_k, best_labels = -1, None, None, None
    for alpha in alpha_grid:
        # unify scale: D = alpha * dist_img + (1-alpha) * dist_meta
        D = alpha * dist_img + (1.0 - alpha) * dist_meta
        # construct affinity for spectral
        sigma = np.std(D) if np.std(D) > 1e-12 else 1.0
        affinity = np.exp(-D / (sigma + 1e-12))

        for k in k_range:
            sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=random_state, n_init=10)
            labels = sc.fit_predict(affinity)
            # Evaluate clustering using MDS embedding of D (distance space) so silhouette is computed consistently
            try:
                n_components = min(10, max(2, img_scaled.shape[1] // 2))
                eval_emb = MDS(n_components=n_components, dissimilarity="precomputed", random_state=random_state).fit_transform(D)
                sil = silhouette_score(eval_emb, labels)
            except Exception:
                sil = -1.0
            if sil > best_sil:
                best_sil, best_alpha, best_k, best_labels = sil, alpha, k, labels

    print(f"[Distribution+MetadataFusion] best_alpha={best_alpha}, best_k={best_k}, silhouette={best_sil:.4f}")

    # optional: visualize using MDS on best D
    if use_mds_for_visual and best_labels is not None:
        try:
            D_best = best_alpha * dist_img + (1.0 - best_alpha) * dist_meta
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state)
            X2 = mds.fit_transform(D_best)
            plt.figure(figsize=(8, 6))
            for cl in np.unique(best_labels):
                idx = best_labels == cl
                plt.scatter(X2[idx, 0], X2[idx, 1], label=f"Cluster {cl}", alpha=0.7)
            plt.legend()
            plt.title(f"Distribution+Metadata Fusion alpha={best_alpha}, k={best_k}")
            plt.savefig(os.path.join("results", "distancefusion_plot.png"), dpi=300, bbox_inches="tight")
            plt.show()
        except Exception as e:
            print("[Warning] MDS visualization failed:", e)

    # return mapping
    return {cid: int(best_labels[cid]) for cid in range(num_clients)}
