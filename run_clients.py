# run_clients_fedprox.py (Adaptive weighting + hierarchical agg + robustness)
import os
import json
import random
import shutil
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import num_clients, num_rounds, is_cluster, num_labels
from utils import generate_and_save_dirichlet_partitioned_data, get_partitioned_data
from cluster_emb import cluster_clients_with_metadata_ratio
from model import CNN

import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy.fedavg import FedAvg
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays

# -------------------
# FedProx ハイパーパラメータ
FEDPROX_MU = 0.0
# Adaptive weighting hyperparams
AWM_LR = 0.1
AWM_ALPHA = 0.4
AWM_BETA = 0.3
AWM_GAMMA = 0.3
# Hierarchical aggregation
USE_HIERARCHICAL = True
NUM_CLUSTERS = 5
# Robustness
USE_ROBUSTNESS = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

RESULTS_BASE_DIR = "results_adaptive"
if os.path.exists(RESULTS_BASE_DIR):
    shutil.rmtree(RESULTS_BASE_DIR)
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

generate_and_save_dirichlet_partitioned_data(num_clients)

if is_cluster:
    client_cluster_map = cluster_clients_with_metadata_ratio(num_clients=num_clients)
else:
    client_cluster_map = {cid: 0 for cid in range(num_clients)}

cluster_list = sorted(set(client_cluster_map.values()))
num_clusters = len(cluster_list)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# AdaptiveWeightingMechanism (簡易だが実用的)
# -----------------------
from scipy.stats import wasserstein_distance
from collections import defaultdict
import math

class AdaptiveWeightingMechanism:
    def __init__(self, alpha=AWM_ALPHA, beta=AWM_BETA, gamma=AWM_GAMMA, history_length=10):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.history_length = history_length
        self.participant_history = defaultdict(list)  # participant_id -> list of past updates (cpu tensors)

    def _class_histogram(self, local_data, num_labels=num_labels):
        """Dataset からクラス頻度ヒストグラムを作る（正規化）"""
        # get_partitioned_data returns a Dataset with .samples or .targets
        counts = np.zeros(num_labels, dtype=float)
        try:
            # torchvision ImageFolder-like
            for _, label in local_data.samples:
                counts[int(label)] += 1
        except Exception:
            # try dataset.targets
            try:
                for label in local_data.targets:
                    counts[int(label)] += 1
            except Exception:
                # fallback: empty
                pass
        total = counts.sum()
        if total > 0:
            return counts / total
        else:
            return np.ones(num_labels, dtype=float) / num_labels

    def compute_diversity_score(self, local_data, participant_id):
        """ここではクラス分布エントロピーを多様性とする"""
        hist = self._class_histogram(local_data)
        # entropy normalized to [0,1]
        ent = -np.sum(hist * np.log(hist + 1e-12))
        max_ent = math.log(len(hist))
        return float(ent / (max_ent + 1e-12))

    def compute_representativeness_score(self, local_data, global_hist):
        """global_hist とローカルヒストの Wass/KL を組み合わせる（簡易実装）"""
        local_hist = self._class_histogram(local_data)
        # wasserstein on class index (1D)
        indices = np.arange(len(local_hist))
        try:
            wdist = wasserstein_distance(indices, indices, u_weights=local_hist, v_weights=global_hist)
            sim_w = 1.0 / (1.0 + wdist)
        except Exception:
            sim_w = 0.0
        # also use cosine similarity as proxy
        num = np.dot(local_hist, global_hist)
        denom = (np.linalg.norm(local_hist) * np.linalg.norm(global_hist) + 1e-12)
        cos_sim = float(num / denom)
        return 0.5 * sim_w + 0.5 * cos_sim

    def compute_consistency_score(self, participant_id, current_update_vec):
        """過去の更新と現在の更新のコサイン類似度平均"""
        hist = self.participant_history.get(participant_id, [])
        if not hist:
            # 初回は高評価
            self.participant_history[participant_id].append(current_update_vec.clone().detach().cpu())
            return 1.0
        sims = []
        cur = current_update_vec.detach().cpu().float()
        for past in hist[-self.history_length:]:
            # ensure same shape
            p = past.view(-1).float()
            if p.numel() != cur.view(-1).numel():
                continue
            cos = torch.nn.functional.cosine_similarity(cur.view(1, -1), p.view(1, -1), dim=1).item()
            sims.append(max(0.0, cos))
        # update history
        hist.append(cur.clone())
        if len(hist) > self.history_length:
            hist.pop(0)
        self.participant_history[participant_id] = hist
        return float(np.mean(sims)) if sims else 1.0

    def compute_quality_score(self, participant_id, local_data, global_hist, current_update_vec):
        d = self.compute_diversity_score(local_data, participant_id)
        r = self.compute_representativeness_score(local_data, global_hist)
        c = self.compute_consistency_score(participant_id, current_update_vec)
        q = float(self.alpha * d + self.beta * r + self.gamma * c)
        return q

    def update_weights(self, participant_qualities, current_weights, learning_rate=AWM_LR):
        mean_quality = float(np.mean(list(participant_qualities.values())))
        updated_weights = {}
        for pid, q in participant_qualities.items():
            cur = current_weights.get(pid, 1.0)
            adj = learning_rate * (q - mean_quality)
            new_w = cur * (1.0 + adj)
            new_w = max(0.0, new_w)
            updated_weights[pid] = new_w
        # normalize to sum=1
        s = sum(updated_weights.values()) + 1e-12
        for pid in updated_weights:
            updated_weights[pid] /= s
        return updated_weights

# -----------------------
# HierarchicalAggregationStrategy (簡易)
# -----------------------
from sklearn.cluster import KMeans

class HierarchicalAggregationStrategy:
    def __init__(self, num_clusters=NUM_CLUSTERS):
        self.num_clusters = num_clusters

    def extract_features_from_hist(self, hist):
        """単純にクラスヒストをそのまま特徴量として使う"""
        return np.array(hist)

    def perform_clustering(self, participant_histograms):
        """
        participant_histograms: dict pid -> hist (1D np.array)
        returns cluster_assignments dict: cluster_id -> [pids]
        """
        pids = list(participant_histograms.keys())
        X = np.vstack([participant_histograms[pid] for pid in pids])
        if len(pids) <= 1:
            return {0: pids}
        k = min(self.num_clusters, len(pids))
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        labels = kmeans.labels_
        clusters = {}
        for pid, lab in zip(pids, labels):
            clusters.setdefault(int(lab), []).append(pid)
        return clusters

    def aggregate_within_cluster(self, cluster_models, cluster_weights):
        """weighted average over state_dict numpy arrays"""
        # cluster_models: pid -> list_of_ndarrays (parameters as numpy arrays)
        # cluster_weights: pid -> scalar weight
        pids = list(cluster_models.keys())
        if not pids:
            return None
        # sum weights
        total_w = sum([cluster_weights.get(pid, 1.0) for pid in pids]) + 1e-12
        # initialize aggregated as zeros array list with same shape as first
        first = cluster_models[pids[0]]
        agg = [np.zeros_like(arr) for arr in first]
        for pid in pids:
            w = cluster_weights.get(pid, 1.0) / total_w
            params = cluster_models[pid]
            for i, arr in enumerate(params):
                agg[i] += w * arr
        return agg

    def aggregate_across_clusters(self, cluster_reps, cluster_qualities):
        """cluster_reps: cid -> params(list of ndarrays)"""
        cids = list(cluster_reps.keys())
        if not cids:
            return None
        total_q = sum([cluster_qualities.get(cid, 1.0) for cid in cids]) + 1e-12
        first = cluster_reps[cids[0]]
        agg = [np.zeros_like(arr) for arr in first]
        for cid in cids:
            q = cluster_qualities.get(cid, 1.0) / total_q
            rep = cluster_reps[cid]
            for i, arr in enumerate(rep):
                agg[i] += q * arr
        return agg

# -----------------------
# RobustnessEnhancementMechanism (簡易)
# -----------------------
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

class RobustnessEnhancementMechanism:
    def __init__(self, contamination=0.1, confidence_threshold=0.5):
        self.contamination = contamination
        self.confidence_threshold = confidence_threshold
        self.update_history = []
        self.scaler = StandardScaler()
        self.iforest = IsolationForest(contamination=contamination, random_state=42)

    def extract_update_features(self, update_params):
        """params: list of numpy arrays"""
        feats = []
        for arr in update_params:
            flat = arr.ravel()
            feats.extend([
                flat.mean(), flat.std(), np.median(flat),
                np.percentile(flat,25), np.percentile(flat,75),
                flat.min(), flat.max(),
                stats.skew(flat), stats.kurtosis(flat)
            ])
        return np.array(feats)

    def detect_statistical_anomaly(self, update_features):
        if len(self.update_history) < 10:
            return 0.0
        hist = np.vstack(self.update_history)
        mean = hist.mean(axis=0); std = hist.std(axis=0) + 1e-12
        z = np.abs((update_features - mean) / std)
        return float(min(1.0, z.max() / 3.0))

    def detect_ml_anomaly(self, update_features):
        if len(self.update_history) < 20:
            return 0.0
        X = np.vstack(self.update_history)
        Xs = self.scaler.fit_transform(X)
        self.iforest.fit(Xs)
        cur = self.scaler.transform(update_features.reshape(1, -1))
        score = self.iforest.decision_function(cur)[0]
        # map to 0-1
        return float(max(0.0, min(1.0, (0.5 - score) / 0.5)))

    def compute_confidence(self, pid, update_features):
        # similarity to recent history
        if len(self.update_history) == 0:
            return 1.0
        recent = np.vstack(self.update_history[-5:])
        sims = [1.0 / (1.0 + np.linalg.norm(update_features - r)) for r in recent]
        mag = np.linalg.norm(update_features)
        mag_score = 1.0 / (1.0 + mag)
        return 0.7 * np.mean(sims) + 0.3 * mag_score

    def adaptive_filtering(self, participant_updates, participant_weights):
        filtered_updates = {}
        filtered_weights = {}
        for pid, params in participant_updates.items():
            feats = self.extract_update_features(params)
            stat = self.detect_statistical_anomaly(feats)
            ml = self.detect_ml_anomaly(feats)
            anomaly = 0.6 * stat + 0.4 * ml
            conf = self.compute_confidence(pid, feats)
            # record history
            self.update_history.append(feats)
            if len(self.update_history) > 200:
                self.update_history.pop(0)
            if anomaly < 0.6 and conf > self.confidence_threshold:
                filtered_updates[pid] = params
                filtered_weights[pid] = participant_weights.get(pid, 1.0)
            elif anomaly < 0.85:
                reduction = max(0.01, 1.0 - anomaly)
                filtered_updates[pid] = params
                filtered_weights[pid] = participant_weights.get(pid, 1.0) * reduction
            else:
                # exclude
                pass
        return filtered_updates, filtered_weights

# -----------------------
# Utility: parameter conversions
# -----------------------
def parameters_to_ndarray_list(parameters):
    """Flower Parameters -> list of numpy arrays"""
    nds = parameters_to_ndarrays(parameters)
    return nds

def ndarray_list_to_parameters(ndlist):
    return ndarrays_to_parameters(ndlist)

# -----------------------
# Custom FedAvg Strategy with adaptive weighting
# -----------------------
class AdaptiveFedAvgStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive = AdaptiveWeightingMechanism()
        self.hier = HierarchicalAggregationStrategy()
        self.robust = RobustnessEnhancementMechanism()
        # initial per-client weights (uniform)
        self.client_weights = {}
        # keep last global params as ndarray list
        self.last_global = None

    def aggregate_fit(self, rnd, results, failures):
        """
        results: list of (client, FitRes) where FitRes.parameters is Parameters
        We'll compute:
         - per-client delta = client_params - last_global
         - quality scores Q_i (uses local dataset via get_partitioned_data)
         - update client_weights based on Q
         - optional robustness filtering
         - optional hierarchical aggregation
        """
        if not results:
            return None, {}

        # convert results into pid -> (ndarray_list, num_examples)
        participant_params = {}
        participant_examples = {}
        # Flower's client object may carry .cid or .node_id in different versions.
        # We rely on FitRes.metadata if present, but to be robust we'll attempt to read client id from client object.
        for client_proxy, fit_res in results:
            # try to find participant id:
            try:
                pid = int(client_proxy.cid)
            except Exception:
                try:
                    pid = int(client_proxy.node_id)
                except Exception:
                    # fallback: use index in results
                    pid = len(participant_params)
            nds = parameters_to_ndarray_list(fit_res.parameters)
            participant_params[pid] = nds
            participant_examples[pid] = int(fit_res.num_examples)

        # initialize global if None
        if self.last_global is None:
            # derive shape from first participant
            first = next(iter(participant_params.values()))
            self.last_global = [np.zeros_like(x) for x in first]

        # compute global class histogram across all clients (for representativeness)
        global_hist = np.zeros(num_labels, dtype=float)
        total_examples = 0
        for pid in participant_params:
            trainset, _ = get_partitioned_data(pid, num_clients)
            hist = self.adaptive._class_histogram(trainset)
            n = sum(getattr(trainset, "targets", []) ) if hasattr(trainset, "targets") else participant_examples.get(pid, 0)
            # more robust: weight by participant_examples
            global_hist += hist * participant_examples.get(pid, 1)
            total_examples += participant_examples.get(pid, 1)
        if total_examples > 0:
            global_hist = global_hist / (total_examples + 1e-12)
        else:
            global_hist = np.ones(num_labels, dtype=float) / num_labels

        # compute per-participant updates (flattened torch vector) for consistency
        participant_updates_vec = {}
        participant_quality = {}
        for pid, params in participant_params.items():
            # delta = params - last_global
            delta = [p - g for p, g in zip(params, self.last_global)]
            # flatten to torch
            flat = torch.from_numpy(np.concatenate([d.ravel() for d in delta])).float()
            participant_updates_vec[pid] = flat
            # get local dataset
            local_data, _ = get_partitioned_data(pid, num_clients)
            # compute Q_i
            q = self.adaptive.compute_quality_score(pid, local_data, global_hist, flat)
            participant_quality[pid] = q

        # update client weights
        # init uniform for unseen
        for pid in participant_params.keys():
            if pid not in self.client_weights:
                self.client_weights[pid] = 1.0 / max(1, len(participant_params))
        updated_weights = self.adaptive.update_weights(participant_quality, self.client_weights)
        # set back
        self.client_weights.update(updated_weights)

        # apply robustness filtering (may reduce weights or drop clients)
        participant_updates = {pid: participant_params[pid] for pid in participant_params}
        if USE_ROBUSTNESS:
            filtered_updates, filtered_weights = self.robust.adaptive_filtering(participant_updates, self.client_weights)
        else:
            filtered_updates = participant_updates
            filtered_weights = self.client_weights

        # if hierarchical aggregation enabled, cluster by local histograms
        if USE_HIERARCHICAL:
            # build histograms map
            part_hists = {}
            for pid in participant_params:
                trainset, _ = get_partitioned_data(pid, num_clients)
                part_hists[pid] = self.adaptive._class_histogram(trainset)
            clusters = self.hier.perform_clustering(part_hists)
            # aggregate within each cluster
            cluster_reps = {}
            cluster_qualities = {}
            for cid, pid_list in clusters.items():
                cluster_models = {pid: filtered_updates[pid] for pid in pid_list if pid in filtered_updates}
                cluster_w = {pid: filtered_weights.get(pid, 0.0) for pid in pid_list}
                if cluster_models:
                    rep = self.hier.aggregate_within_cluster(cluster_models, cluster_w)
                    cluster_reps[cid] = rep
                    cluster_qualities[cid] = float(np.mean([participant_quality.get(pid, 0.0) for pid in pid_list]))
            # aggregate across clusters
            if cluster_reps:
                agg = self.hier.aggregate_across_clusters(cluster_reps, cluster_qualities)
                aggregated_nds = agg
            else:
                # fallback to weighted average over filtered_updates
                total_w = sum(filtered_weights.values()) + 1e-12
                first = next(iter(filtered_updates.values()))
                aggregated_nds = [np.zeros_like(arr) for arr in first]
                for pid, params in filtered_updates.items():
                    w = filtered_weights.get(pid, 0.0) / total_w
                    for i, arr in enumerate(params):
                        aggregated_nds[i] += w * arr
        else:
            # simple weighted average
            total_w = sum(filtered_weights.values()) + 1e-12
            first = next(iter(filtered_updates.values()))
            aggregated_nds = [np.zeros_like(arr) for arr in first]
            for pid, params in filtered_updates.items():
                w = filtered_weights.get(pid, 0.0) / total_w
                for i, arr in enumerate(params):
                    aggregated_nds[i] += w * arr

        # update last_global
        self.last_global = [nd.copy() for nd in aggregated_nds]

        # return Parameters
        aggregated_parameters = ndarray_list_to_parameters(aggregated_nds)
        # optional: custom metrics (we return participant qualities for logging)
        custom_metrics = {"participant_qualities": participant_quality, "updated_weights": self.client_weights}
        return aggregated_parameters, custom_metrics

# -----------------------
# FL Wrapping: keep same FL client implementation (no change here)
# -----------------------
from flwr.client import NumPyClient

class FLClient(NumPyClient):
    def __init__(self, cid, active_cids, mu=FEDPROX_MU):
        self.cid = int(cid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN(num_classes=num_labels).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.mu = float(mu)

        if self.cid in active_cids:
            trainset, testset = get_partitioned_data(self.cid, num_clients)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=32)
        else:
            self.trainloader = []
            self.testloader = []

    def log(self, msg):
        print(f"[Client {self.cid}] {msg}")

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, val in zip(self.model.parameters(), parameters):
            p.data = torch.from_numpy(val).to(self.device).to(torch.float32)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_params = [p.detach().clone() for p in self.model.parameters()]
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.8)

        self.model.train()
        prev_loss = float('inf')
        EPOCHS = 1
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                if self.mu > 0:
                    prox_term = 0.0
                    for p, g in zip(self.model.parameters(), global_params):
                        prox_term += torch.sum((p - g.to(self.device)) ** 2)
                    loss = loss + (self.mu / 2.0) * prox_term
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            if abs(prev_loss - running_loss) < 1e-3:
                break
            prev_loss = running_loss

        self.log(f"Finished local training (final loss={prev_loss:.4f}, mu={self.mu})")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        avg_loss = total_loss / len(self.testloader.dataset) if len(self.testloader.dataset) > 0 else 0.0
        accuracy = correct / len(self.testloader.dataset) if len(self.testloader.dataset) > 0 else 0.0
        self.log(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        return avg_loss, len(self.testloader.dataset), {"accuracy": accuracy, "loss": avg_loss}

# -----------------------
# Start simulation (use AdaptiveFedAvgStrategy)
# -----------------------
for cluster_id in range(num_clusters):
    selected_cids = [cid for cid, clid in client_cluster_map.items() if clid == cluster_id]
    print(f"\n--- cluster {cluster_id} simulation start ---")
    cluster_results = {}

    def aggregate_metrics(results):
        # This aggregator is only for evaluation metrics reporting; we will not use it for parameter aggregation.
        print("\n📊 This round client evals:")
        weighted_sum_acc = 0.0
        total_weight = 0.0
        total_examples = 0
        for i, (num_examples, metrics) in enumerate(results):
            acc = metrics["accuracy"]
            weight = num_examples
            weighted_sum_acc += acc * weight
            total_weight += weight
            total_examples += num_examples
            print(f"  Client {i}: Acc={acc*100:.2f}%, Samples={num_examples}")
        avg_accuracy = weighted_sum_acc / (total_weight + 1e-12)
        cluster_results["accuracy"] = avg_accuracy
        cluster_results["samples"] = total_examples
        print(f"Cluster {cluster_id} approx acc: {avg_accuracy*100:.2f}%")
        return {"accuracy": avg_accuracy}

    strategy = AdaptiveFedAvgStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(selected_cids),
        min_available_clients=len(selected_cids),
        min_evaluate_clients=len(selected_cids),
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )

    from flwr.common import Context
    client_cache = {}

    def client_fn(context: Context):
        cid_int = context.node_config.get("partition-id", context.node_id)
        idx = int(cid_int)
        real_cid = selected_cids[idx]
        if real_cid not in client_cache:
            client_cache[real_cid] = FLClient(real_cid, selected_cids, mu=FEDPROX_MU).to_client()
        return client_cache[real_cid]

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(selected_cids),
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1.0},
    )

    final_cluster_metrics = {
        "final_accuracy": cluster_results.get("accuracy", 0.0),
        "samples": cluster_results.get("samples", 0),
    }
    # save per-cluster results
    outpath = os.path.join(RESULTS_BASE_DIR, f"cluster_{cluster_id}_summary.json")
    with open(outpath, "w") as f:
        json.dump(final_cluster_metrics, f, indent=2)

print("All clusters done. Results in", RESULTS_BASE_DIR)
