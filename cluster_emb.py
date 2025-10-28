# cluster_emb.py
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.manifold import MDS, TSNE

import matplotlib.pyplot as plt

from utils import get_partitioned_data
from config import num_clients, backbone_name

# -----------------------
# params: adjustable grid
# -----------------------
params = {
    # fusion method
    'method': 'distance',             # 'concat' or 'distance'
    'cluster_method': 'spectral',     # 'kmeans' / 'hierarchical' / 'spectral' / 'dbscan'
    'k_range': range(3, 10),          # candidate cluster k
    'alpha_grid': np.linspace(0.2, 0.8, 7).tolist(),  # fusion alpha grid (image weight)
    'metadata_weight': None,          # None -> auto scale
    'pca_meta_dim': 10,               # compress metadata ratios to at most this dim (auto-adjusted)
    'use_mds_for_visual': True,
    'mds_dim': 2,
    'random_state': 42,
    # embedding MDS/PCA dims for silhouette eval
    'eval_embed_dim': 10,
}

# reproducibility
SEED = params['random_state']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------
# helper: extract embedding
# -----------------------
def extract_client_embedding_for_batch(model, x, device):
    """
    Return per-sample embedding (numpy array shape (B, D)).
    Supports two model types:
      - ResNet-like with attribute `feature_extractor` and avgpool (our model.py)
      - small CNN with method `_forward_small` that returns conv feature map
    """
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        if hasattr(model, "feature_extractor"):
            feats = model.feature_extractor(x)  # e.g., (B, 512, 1, 1) or (B, C, H, W)
            # If final spatial is >1, apply adaptive pool
            if feats.dim() == 4:
                feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))
                feats = feats.view(feats.size(0), -1)
            else:
                feats = feats.view(feats.size(0), -1)
            emb = feats.cpu().numpy()
            return emb
        elif hasattr(model, "_forward_small"):
            feats = model._forward_small(x)  # conv output, e.g. (B, C, H, W)
            feats = feats.view(feats.size(0), -1)
            emb = feats.cpu().numpy()
            return emb
        else:
            # fallback: try forward but remove final classifier if possible
            out = model(x)
            if out.dim() == 2:
                return out.cpu().numpy()
            else:
                return out.view(out.size(0), -1).cpu().numpy()

def extract_features(client_id, model, device, batch_size=32):
    """
    Extract mean embedding per client.
    Returns: numpy array shape (embedding_dim,)
    """
    dataset, _ = get_partitioned_data(client_id, num_clients)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    for x, _ in loader:
        emb = extract_client_embedding_for_batch(model, x, device)
        embeddings.append(emb)
    if len(embeddings) == 0:
        # empty client guard
        return np.zeros((1, 1)).reshape(-1)
    embeddings = np.vstack(embeddings)
    mean_emb = embeddings.mean(axis=0)
    return mean_emb

# -----------------------
# utility: safe perplexity
# -----------------------
def safe_tsne_perplexity(n_samples):
    # recommended: 5 <= perplexity <= 50 and < n_samples/3
    if n_samples < 15:
        return 3
    return min(30, max(5, n_samples // 3))

# -----------------------
# main clustering function
# -----------------------
def cluster_clients_with_metadata_ratio(num_clients, feature_extractor=None):
    """
    Returns dict: {client_id: cluster_id}
    Uses global params dict above.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load or use provided feature_extractor
    if feature_extractor is None:
        from model import CNN
        # create model consistent with backbone_name
        if backbone_name == "resnet18":
            model = CNN(num_classes=38, backbone="resnet18", pretrained=True)
        else:
            model = CNN(num_classes=38, backbone="small", pretrained=False)
    else:
        model = feature_extractor
    model.to(device)

    # -----------------------------
    # 1) extract image embeddings per client
    # -----------------------------
    print("[Step] extracting image embeddings for each client...")
    client_features = []
    for cid in range(num_clients):
        emb = extract_features(cid, model, device, batch_size=32)
        client_features.append(emb)
    client_features = np.vstack(client_features)  # shape (N_clients, feat_dim)
    print(f" extracted image embeddings shape: {client_features.shape}")

    # -----------------------------
    # 2) build metadata ratio vectors per client
    # -----------------------------
    print("[Step] building metadata ratio vectors per client...")
    client_label_stats = []
    all_crops, all_diseases = set(), set()
    for cid in range(num_clients):
        dataset, _ = get_partitioned_data(cid, num_clients)
        class_names = dataset.classes
        label_counts = defaultdict(int)
        # iterate dataset samples safely
        for path, label in dataset.samples:
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
    print(f" metadata ratios shape: {metadata_ratios.shape}")

    # -----------------------------
    # 3) PCA compress metadata (avoid very high dim)
    # -----------------------------
    pca_meta_dim = min(params['pca_meta_dim'], metadata_ratios.shape[1], max(1, metadata_ratios.shape[0] - 1))
    if metadata_ratios.shape[1] > pca_meta_dim:
        pca = PCA(n_components=pca_meta_dim, random_state=SEED)
        meta_reduced = pca.fit_transform(metadata_ratios)
        print(f" metadata PCA reduced to {meta_reduced.shape[1]} dims (explained {pca.explained_variance_ratio_.sum():.3f})")
    else:
        meta_reduced = metadata_ratios

    # -----------------------------
    # 4) compute auto metadata_weight (scale)
    # -----------------------------
    img_var = np.var(client_features)
    meta_var = np.var(meta_reduced)
    metadata_weight = params.get('metadata_weight', None)
    if metadata_weight is None:
        metadata_weight = float(np.sqrt((img_var + 1e-12) / (meta_var + 1e-12)))
        metadata_weight = float(np.clip(metadata_weight, 1e-6, 1e6))
    print(f"[Auto] metadata_weight = {metadata_weight:.6g}")

    # -----------------------------
    # 5) standardize each modality
    # -----------------------------
    img_scaled = StandardScaler().fit_transform(client_features)
    meta_scaled = StandardScaler().fit_transform(meta_reduced * metadata_weight)

    # -----------------------------
    # 6) method branches
    # -----------------------------
    method = params.get('method', 'distance')
    cluster_method = params.get('cluster_method', 'spectral')
    k_range = list(params.get('k_range', range(2, 7)))
    alpha_grid = list(params.get('alpha_grid', [0.0, 0.25, 0.5, 0.75, 1.0]))
    use_mds_for_visual = params.get('use_mds_for_visual', True)
    random_state = params.get('random_state', 42)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    metrics_out = {
        "method": method,
        "cluster_method": cluster_method,
        "metadata_weight": metadata_weight,
        "num_clients": int(num_clients),
        "crop_list": crop_list,
        "disease_list": disease_list,
    }

    # -----------------------------
    # concat method (simple)
    # -----------------------------
    if method == "concat":
        X = np.hstack([img_scaled, meta_scaled])
        print(f"[Concat] combined feature dim: {X.shape}")
        if cluster_method == "kmeans":
            best_k, best_labels, best_score = None, None, -np.inf
            scores_by_k = {}
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20).fit(X)
                labels = kmeans.labels_
                sil = silhouette_score(X, labels)
                ch = calinski_harabasz_score(X, labels)
                db = davies_bouldin_score(X, labels)
                combined_score = sil + ch / 1000.0 - db / 10.0
                scores_by_k[int(k)] = {"silhouette": float(sil), "ch": float(ch), "db": float(db), "combined": float(combined_score)}
                if combined_score > best_score:
                    best_score = combined_score
                    best_k = int(k)
                    best_labels = labels
            metrics_out.update({"k_candidates": k_range, "k_scores": scores_by_k, "k_opt": best_k})
            metrics_out.update({"silhouette": float(sil), "calinski_harabasz": float(ch), "davies_bouldin": float(db)})
            with open(os.path.join(results_dir, "cluster_metrics.json"), "w") as f:
                json.dump(metrics_out, f, indent=2)
            visualize_clusters(X, best_labels := np.array(best_labels), title=f"Concat Clusters (k={best_k})")
            return {cid: int(best_labels[cid]) for cid in range(num_clients)}
        else:
            # fallback hierarchical / dbscan
            if cluster_method == "hierarchical":
                labels = AgglomerativeClustering(n_clusters=max(k_range), linkage='ward').fit_predict(X)
            elif cluster_method == "dbscan":
                labels = DBSCAN(metric='euclidean', eps=0.5, min_samples=5).fit_predict(X)
            else:
                raise ValueError("Unsupported cluster_method for concat")
            visualize_clusters(X, labels, title=f"Concat Clusters ({cluster_method})")
            return {cid: int(labels[cid]) for cid in range(num_clients)}

    # -----------------------------
    # distance fusion branch
    # -----------------------------
    elif method == "distance":
        # compute pairwise distances
        d_img = pairwise_distances(img_scaled, metric="euclidean")
        d_meta = pairwise_distances(meta_scaled, metric="euclidean")

        # normalize distances to [0,1] to avoid scale domination
        if d_img.max() > 0:
            d_img = d_img / (d_img.max() + 1e-12)
        if d_meta.max() > 0:
            d_meta = d_meta / (d_meta.max() + 1e-12)

        best_alpha = None
        best_k = None
        best_labels = None
        best_silhouette = -np.inf
        k_scores_by_alpha = {}

        for alpha in alpha_grid:
            D = alpha * d_img + (1.0 - alpha) * d_meta
            # compute sigma for gaussian kernel using median heuristic on D values
            flat = D[np.triu_indices_from(D, k=1)]
            if flat.size == 0:
                sigma = 1.0
            else:
                sigma = np.median(flat)
                if sigma <= 0:
                    sigma = np.std(flat) if np.std(flat) > 0 else 1.0

            # gaussian affinity (note squared distance)
            A = np.exp(- (D ** 2) / (2.0 * (sigma ** 2 + 1e-12)))

            scores_k = {}
            for k in k_range:
                labels = None
                try:
                    if cluster_method == "spectral":
                        sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=random_state, n_init=10)
                        labels = sc.fit_predict(A)
                    elif cluster_method == "hierarchical":
                        # Agglomerative on distance matrix: convert to linkage via MDS embedding
                        emb = MDS(n_components=min(params['eval_embed_dim'], max(2, img_scaled.shape[1] // 2)),
                                  dissimilarity="precomputed", random_state=random_state).fit_transform(D)
                        labels = AgglomerativeClustering(n_clusters=k, linkage='average').fit_predict(emb)
                    elif cluster_method == "dbscan":
                        labels = DBSCAN(metric='precomputed', eps=0.5, min_samples=5).fit_predict(D)
                    elif cluster_method == "kmeans":
                        # kmeans on concatenated embeddings as baseline
                        emb_concat = np.hstack([img_scaled, meta_scaled])
                        labels = KMeans(n_clusters=k, random_state=random_state, n_init=20).fit_predict(emb_concat)
                    else:
                        raise ValueError("Unsupported cluster_method")
                except Exception as e:
                    print(f"[Warning] clustering failed for alpha={alpha}, k={k}, err={e}")
                    labels = np.array([-1] * num_clients)

                # evaluate using embedding for silhouette: use MDS embedding from D
                try:
                    eval_emb = MDS(n_components=min(params['eval_embed_dim'], max(2, img_scaled.shape[1] // 2)),
                                   dissimilarity="precomputed", random_state=random_state).fit_transform(D)
                    sil = silhouette_score(eval_emb, labels)
                    ch = calinski_harabasz_score(eval_emb, labels) if len(np.unique(labels)) > 1 else -1.0
                    db = davies_bouldin_score(eval_emb, labels) if len(np.unique(labels)) > 1 else -1.0
                except Exception:
                    sil, ch, db = -1.0, -1.0, -1.0

                combined_score = float(sil + (ch / 1000.0 if ch > 0 else 0.0) - (db / 10.0 if db > 0 else 0.0))
                scores_k[int(k)] = {"silhouette": float(sil), "calinski_harabasz": float(ch), "davies_bouldin": float(db), "combined": combined_score}
                # choose best by silhouette primarily
                if sil > best_silhouette:
                    best_silhouette = sil
                    best_alpha = float(alpha)
                    best_k = int(k)
                    best_labels = labels

            k_scores_by_alpha[float(alpha)] = scores_k

        metrics_out.update({
            "alpha_grid": alpha_grid,
            "best_alpha": best_alpha,
            "k_candidates": k_range,
            "k_scores_by_alpha": k_scores_by_alpha,
            "k_opt": int(best_k) if best_k is not None else None,
            "silhouette": float(best_silhouette)
        })
        with open(os.path.join(results_dir, "cluster_metrics.json"), "w") as f:
            json.dump(metrics_out, f, indent=2)

        print(f"[DistanceFusion] best_alpha={best_alpha}, k={best_k}, silhouette={best_silhouette:.4f}")

        # visualization using MDS
        if use_mds_for_visual and best_labels is not None:
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

# -----------------------
# visualization helper
# -----------------------
def visualize_clusters(features, cluster_ids, title="Client Feature Clusters (t-SNE 2D Projection)"):
    n = features.shape[0]
    perp = safe_tsne_perplexity(n)
    tsne = TSNE(n_components=2, random_state=params['random_state'], perplexity=perp)
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
