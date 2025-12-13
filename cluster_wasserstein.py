# cluster_emb.py (params fully unified; distance/kernel fusion selectable)
import json
import os
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS, TSNE
import time
import concurrent.futures
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from utils import get_partitioned_data
from config import num_clients

# ============================================================
# ✅ パラメータ 100% 一元管理
# ============================================================
params = {
    # clustering
    "method": "distance",
    "cluster_method": "spectral",
    "k_range": list(range(5, 10)),
    "alpha_grid": [1.0],  # include endpoints
    "use_mds_for_visual": True,
    "mds_dim": 2,
    "random_state": 5,

    # fusion selection: "distance" or "kernel"
    "fusion_mode": "distance",

    # distribution similarity (Sliced WD)
    "wasserstein_mode": "sliced",
    "initial_L": 64,
    "refine_L": 192,
    "refine_topk": 6,
    "subsample": None,
    "pca_dim": 64,        # PCA次元 統一
    "n_jobs": 8,

    # metadata scaling
    "metadata_weight": None,
    "max_cluster_size": None,  # None: no restriction, or int to limit largest cluster
}

# fix seeds
SEED = params["random_state"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# Sliced WD utilities
# ============================================================
def _proj_and_sort_for_projection_local(v, client_features_list, subsample=None, rng=None):
    sorted_vals = []
    for Xi in client_features_list:
        if subsample is not None and Xi.shape[0] > subsample:
            idx = rng.choice(Xi.shape[0], size=subsample, replace=False)
            Xi_use = Xi[idx]
        else:
            Xi_use = Xi
        # v is shape (d,), Xi_use (n, d)
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


def compute_sliced_wasserstein_matrix(
    client_features_list,
    initial_L,
    refine_L,
    refine_topk,
    subsample,
    pca_dim,
    n_jobs,
    seed,
):
    """
    二段階 Sliced Wasserstein (coarse -> refine)。
    入力 client_features_list は各クライアントの (n_i, D_orig) 配列のリスト。
    戻り値:
      final_distance_matrix (n_clients, n_clients),
      client_features_list_pca: PCA後の各クライアント特徴リスト（対応する平均ベクトルはこれから作る）
    """
    rng = np.random.RandomState(seed)

    # PCA space unify
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

    # coarse direction
    Vs_coarse = rng.normal(size=(initial_L, d))
    Vs_coarse /= np.linalg.norm(Vs_coarse, axis=1, keepdims=True) + 1e-12

    start = time.time()
    sorted_vals_coarse = [None] * initial_L

    def job_c(k):
        r = np.random.RandomState(seed + 1000 + k)
        return _proj_and_sort_for_projection_local(Vs_coarse[k], client_features_list, subsample, r)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_jobs, initial_L)) as exe:
        futures = {exe.submit(job_c, k): k for k in range(initial_L)}
        for f in concurrent.futures.as_completed(futures):
            sorted_vals_coarse[futures[f]] = f.result()

    dist_coarse = np.zeros((n_clients, n_clients))
    for k in range(initial_L):
        dist_coarse += _sliced_wasserstein_pairwise_from_proj_local(sorted_vals_coarse[k])
    dist_coarse /= float(initial_L)
    print(f"[coarse SWD] {time.time() - start:.1f}s")

    # pick refine candidate pairs (topk per client)
    topk = max(1, refine_topk)
    candidates = set()
    for i in range(n_clients):
        tmp = dist_coarse[i].copy()
        tmp[i] = np.inf
        idx = np.argpartition(tmp, topk)[:topk]
        for j in idx:
            a, b = sorted((i, j))
            candidates.add((a, b))
    candidates = sorted(candidates)

    # refine directions
    Vs_refine = rng.normal(size=(refine_L, d))
    Vs_refine /= np.linalg.norm(Vs_refine, axis=1, keepdims=True) + 1e-12

    start = time.time()
    sorted_vals_ref = [None] * refine_L

    def job_r(k):
        r = np.random.RandomState(seed + 2000 + k)
        return _proj_and_sort_for_projection_local(Vs_refine[k], client_features_list, subsample, r)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_jobs, refine_L)) as exe:
        futures = {exe.submit(job_r, k): k for k in range(refine_L)}
        for f in concurrent.futures.as_completed(futures):
            sorted_vals_ref[futures[f]] = f.result()

    refined = {}
    for a, b in candidates:
        acc = 0.0
        for k in range(refine_L):
            acc += wasserstein_distance(sorted_vals_ref[k][a], sorted_vals_ref[k][b])
        refined[(a, b)] = acc / float(refine_L)

    final = dist_coarse.copy()
    for (a, b), w in refined.items():
        final[a, b] = final[b, a] = (final[a, b] + w) / 2.0
    print(f"[refine SWD] {time.time() - start:.1f}s  candidate={len(candidates)}")

    return final, client_features_list


# ============================================================
# feature extraction
# ============================================================
def extract_features(client_id, model, device):
    dataset, _ = get_partitioned_data(client_id, num_clients)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    feats = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            feats.append(model(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


# ============================================================
# main clustering
# ============================================================
def cluster_clients_with_metadata_ratio(num_clients, feature_extractor=None, use_distribution=True):
    """
    メイン関数。params に従って
      - Sliced Wasserstein (または 1D legacy) で dist_img を作成
      - メタデータ比率ベクトル作成 -> 標準化 -> weight 適用
      - fusion_mode に従って distance fusion または kernel fusion を適用
      - spectral clustering で最良 alpha/k を探索（silhouette で判定）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature_extractor is None:
        from model import CNN
        model = CNN(num_classes=38)
        model.fc2 = nn.Identity()
    else:
        model = feature_extractor

    model.to(device)

    # extract all sample features (raw)
    feats_raw = []
    for cid in range(num_clients):
        feats_raw.append(extract_features(cid, model, device))

    # distribution similarity (sliced)
    if use_distribution and params["wasserstein_mode"] == "sliced":
        dist_img, feats_list = compute_sliced_wasserstein_matrix(
            feats_raw,
            params["initial_L"],
            params["refine_L"],
            params["refine_topk"],
            params["subsample"],
            params["pca_dim"],
            params["n_jobs"],
            params["random_state"],
        )
        # feats_list are PCA-space per-sample arrays; get per-client mean in PCA space
        feats_mean = np.vstack([f.mean(axis=0) for f in feats_list])
    else:
        raise NotImplementedError("Only 'sliced' wasserstein_mode is implemented in this function.")

    # metadata ratio (crop + disease)
    stats = []
    crops, dis = set(), set()
    for cid in range(num_clients):
        dataset, _ = get_partitioned_data(cid, num_clients)
        label_counts = defaultdict(int)
        for _, lab in dataset:
            label_counts[dataset.classes[lab]] += 1
        stats.append(label_counts)
        for cname in dataset.classes:
            if "___" in cname:
                crop, disease = cname.split("___", 1)
            else:
                crop, disease = cname, "Unknown"
            crops.add(crop)
            dis.add(disease)

    crops = sorted(crops)
    dis = sorted(dis)

    def get_vec(st):
        cc = defaultdict(int)
        dd = defaultdict(int)
        tot = sum(st.values()) or 1
        for cname, cnt in st.items():
            if "___" in cname:
                crop, ds = cname.split("___", 1)
            else:
                crop, ds = cname, "Unknown"
            cc[crop] += cnt
            dd[ds] += cnt
        return np.concatenate([
            np.array([cc[c] / tot for c in crops]),
            np.array([dd[d] / tot for d in dis]),
        ])

# === Step 1: Get features ===
    meta = np.vstack([get_vec(s) for s in stats])

    # === Step 2: Standardize both feature views ===
    scaler_meta = StandardScaler()
    scaler_img  = StandardScaler()
    meta = scaler_meta.fit_transform(meta)
    img  = scaler_img.fit_transform(feats_mean)

    # === Step 3: Decide metadata weight only for distance fusion ===
    img_var  = np.mean(np.var(img, axis=0))
    meta_var = np.mean(np.var(meta, axis=0))

    if params.get("fusion_mode", "distance") == "distance":
        w = params.get("metadata_weight", None)
        if w is None:
            w = np.sqrt((img_var + 1e-12) / (meta_var + 1e-12))
            w = float(np.clip(w, 1e-3, 1e3))
        meta_scaled = meta * w
        print(f"[Auto] metadata_weight (distance fusion) = {w:.4f}")
    else:
        # kernel fusionでは特徴量の重みは距離レベルでなくkernelに含まれるので w適用しない
        meta_scaled = meta
        print("[Info] metadata weight disabled for kernel fusion")

    # === Step 4: Distance computation ===
    eps = 1e-12
    dist_img  = dist_img / (np.std(dist_img) + eps)
    dist_meta = pairwise_distances(meta_scaled, metric="euclidean")
    dist_meta = dist_meta / (np.std(dist_meta) + eps)

    fusion_mode = params.get("fusion_mode", "distance")
    print(f"[Info] fusion_mode = {fusion_mode}, alpha_grid = {params['alpha_grid']}")

    # === Step 5: Kernel precompute (only if needed) ===
    if fusion_mode == "kernel":
        sigma_img  = np.std(dist_img) if np.std(dist_img) > 1e-12 else 1.0
        sigma_meta = np.std(dist_meta) if np.std(dist_meta) > 1e-12 else 1.0

        A_img  = np.exp(-(dist_img**2)  / (2 * sigma_img**2  + eps))
        A_meta = np.exp(-(dist_meta**2) / (2 * sigma_meta**2 + eps))

        A_img  = (A_img  + A_img.T ) / 2
        A_meta = (A_meta + A_meta.T) / 2

    best_sil = -np.inf
    best = None
    max_allowed = params.get("max_cluster_size", None)


    # === Step 6: Grid search over α and cluster count ===
    for a in params["alpha_grid"]:

        if fusion_mode == "distance":
            D = a * dist_img + (1 - a) * dist_meta
            sigma = np.std(D) if np.std(D) > 1e-12 else 1.0
            A = np.exp(-(D**2) / (2 * sigma**2 + eps))
            A = (A + A.T) / 2

        elif fusion_mode == "kernel":
            A = a * A_img + (1 - a) * A_meta
            A = (A + A.T) / 2
            D = 1 - (A - A.min()) / (A.max() - A.min() + eps)

        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        # === Step 7: Search cluster numbers ===
        for k in params["k_range"]:
            try:
                sc = SpectralClustering(
                    n_clusters=k, affinity="precomputed", 
                    random_state=params["random_state"], n_init=10
                )
                labels = sc.fit_predict(A)
            except Exception:
                continue

            try:
                sil = silhouette_score(D, labels, metric="precomputed")
            except Exception:
                sil = -1.0
            if max_allowed is not None:
                _, counts = np.unique(labels, return_counts=True)
                max_size = np.max(counts)
                if max_size > max_allowed:
                    continue

            if sil > best_sil:
                best_sil = sil
                best = {"alpha": a, "k": k, "labels": labels.copy(), "sil": sil}
                print(f"[BestUpdate] α={a}, k={k}, sil={sil:.4f}")

    print(f"[RESULT] fusion={fusion_mode}, alpha={best['alpha']}, k={best['k']}, silhouette={best['sil']:.4f}")

    # optional: visualize using MDS on D (distance-like)
    if params["use_mds_for_visual"]:
        try:
            D_vis = None
            if params["fusion_mode"] == "distance":
                D_vis = best["alpha"] * dist_img_n + (1 - best["alpha"]) * dist_meta_n
            else:
                # for kernel, turn affinity into distance proxy
                A_best = best["alpha"] * A_img + (1 - best["alpha"]) * A_meta
                D_vis = 1.0 - (A_best - A_best.min()) / (A_best.max() - A_best.min() + eps)
            X = MDS(n_components=2, dissimilarity="precomputed", random_state=params["random_state"]).fit_transform(D_vis)
            plt.figure(figsize=(8, 6))
            for cl in np.unique(best["labels"]):
                idx = best["labels"] == cl
                plt.scatter(X[idx, 0], X[idx, 1], label=f"Cluster{cl}", alpha=0.7)
            plt.legend()
            plt.title(f"Fusion {params['fusion_mode']} alpha={best['alpha']}, k={best['k']}")
            plt.savefig("FusionCluster_MDS.png", dpi=300)
            plt.close()
        except Exception as e:
            print("[Warning] MDS visualization failed:", e)

    return {cid: int(best["labels"][cid]) for cid in range(num_clients)}
