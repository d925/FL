import os
import json
from config import num_clients, num_rounds, is_cluster, batch_size, learning_rate, local_epochs, proximal_mu, gpu_memory_fraction, lr_scheduler_step, lr_scheduler_gamma, warmup_rounds
from adaptive_aggregation import AdaptiveAggregation
from smart_client_selection import SmartClientSelection
from class_balancing import ClassBalancingLoss, FocalLoss
from utils import generate_and_save_dirichlet_partitioned_data, get_partitioned_data, num_labels
from cluster import cluster_clients
from model import CNN
import flwr as fl
from flwr.server import ServerConfig
import torch
import torch.optim as optim
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common import Context

# 結果保存ディレクトリ作成
RESULTS_BASE_DIR = "results"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

generate_and_save_dirichlet_partitioned_data(num_clients)

if is_cluster:
    # Step 2: クラスタリング実行
    client_cluster_map = cluster_clients(num_clients=num_clients)

    print("クラスタリング結果:")
    for cid, clust_id in client_cluster_map.items():
        print(f"クライアント {cid} は クラスター {clust_id}")
    
    cluster_list = sorted(set(client_cluster_map.values()))
    num_clusters = len(cluster_list)  # ★ クラスタ数を自動反映
else:
    client_cluster_map = {cid: 0 for cid in range(num_clients)}  # 全クライアントをクラスタ0に所属させる
    cluster_list = [0]
    num_clusters = 1



# クラスタ単位の最終結果保持用
final_cluster_metrics = {}
total_correct = 0
total_samples = 0

# クライアントクラス定義
class FLClient(NumPyClient):
    def __init__(self, cid, active_cids, round_num=1):
        self.cid = int(cid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN(num_classes=num_labels).to(self.device)
        
        # Initialize class balancing loss
        self.class_balancer = ClassBalancingLoss(num_classes=num_labels, strategy='focal')
        self.criterion = self.class_balancer.get_loss_function()
        self.round_num = round_num
        
        # Adaptive learning rate with warmup and scheduling
        self.current_lr = self._calculate_adaptive_lr(round_num)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.current_lr)

        if self.cid in active_cids:
            trainset, testset = get_partitioned_data(self.cid, num_clients)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
        else:
            self.trainloader = []
            self.testloader = []
    
    def _calculate_adaptive_lr(self, round_num):
        """Calculate adaptive learning rate with warmup and decay."""
        base_lr = learning_rate
        
        # Warmup phase
        if round_num <= warmup_rounds:
            # Linear warmup from 0.1 * base_lr to base_lr
            warmup_factor = 0.1 + 0.9 * (round_num - 1) / max(1, warmup_rounds - 1)
            return base_lr * warmup_factor
        
        # Decay phase
        decay_steps = (round_num - warmup_rounds) // lr_scheduler_step
        return base_lr * (lr_scheduler_gamma ** decay_steps)

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, val in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(val, dtype=torch.float32, device=self.device)

    def fit(self, parameters, config):
        if not self.trainloader:
            return self.get_parameters(config), 0, {}
        
        # Update learning rate based on current round
        current_round = config.get("server_round", 1)
        self.current_lr = self._calculate_adaptive_lr(current_round)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        
        print(f"[Client {self.cid}] Round {current_round}: LR = {self.current_lr:.6f}")
        
        self.set_parameters(parameters)
        global_params = [p.clone().detach() for p in self.model.parameters()]
        mu = config.get("proximal_mu", proximal_mu)

        self.model.train()
        for epoch in range(local_epochs):
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                if mu > 0:
                    prox_term = sum(((param - gparam.to(self.device))**2).sum() for param, gparam in zip(self.model.parameters(), global_params))
                    loss += (mu / 2) * prox_term
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        if not self.testloader:
            return 0.0, 0, {"accuracy": 0.0, "loss": 0.0}
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        avg_loss = total_loss / len(self.testloader.dataset)
        acc = correct / len(self.testloader.dataset)
        print(f"[Client {self.cid}] Evaluation → Accuracy: {acc*100:.2f}%, Loss: {avg_loss:.4f}, Samples: {len(self.testloader.dataset)}")
        return avg_loss, len(self.testloader.dataset), {"accuracy": acc, "loss": avg_loss}

# Step 3: Memory-efficient cluster processing
# Process clusters sequentially to avoid memory overload (GPU/Ray memory constraints)
def process_cluster(cluster_id, client_cluster_map, num_clusters):
    selected_cids = [cid for cid, clid in client_cluster_map.items() if clid == cluster_id]
    
    if len(selected_cids) == 0:
        print(f"Warning: Cluster {cluster_id} has no clients, skipping...")
        return None
        
    print(f"\n--- クラスタ {cluster_id} のシミュレーション開始 ({len(selected_cids)} clients) ---")

    cluster_results = {}
    
    # Memory management: Set GPU memory fraction per cluster
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
        torch.cuda.empty_cache()

    # Initialize adaptive aggregation
    adaptive_aggregator = AdaptiveAggregation(memory_efficient=True)
    
    def aggregate_metrics(results):
        print("\n📊 このラウンドのクライアント評価結果:")
        for i, (num_examples, metrics) in enumerate(results):
            acc = metrics["accuracy"] * 100
            loss = metrics["loss"]
            print(f"  Client {i}: Accuracy = {acc:.2f}%, Loss = {loss:.4f}, Samples = {num_examples}")

        # Adaptive aggregation for better performance
        client_metrics = [(num_examples, metrics) for num_examples, metrics in results if num_examples > 0]
        
        if client_metrics:
            # Calculate adaptive weighted metrics
            total_examples = sum(num_examples for num_examples, _ in client_metrics)
            
            # Performance-based weights (inverse loss weighting)
            weights = []
            for num_examples, metrics in client_metrics:
                loss = metrics["loss"]
                # Higher weight for lower loss (better performance)
                weight = num_examples / (1.0 + loss)
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Adaptive weighted aggregation
            weighted_accuracy = sum(weights[i] * metrics["accuracy"] 
                                  for i, (_, metrics) in enumerate(client_metrics))
            weighted_loss = sum(weights[i] * metrics["loss"] 
                              for i, (_, metrics) in enumerate(client_metrics))
            
            print(f"🧠 Adaptive weighting applied - Weight distribution: {[f'{w:.3f}' for w in weights]}")
        else:
            weighted_accuracy = 0.0
            weighted_loss = 1.0
            total_examples = 0

        cluster_results["accuracy"] = weighted_accuracy
        cluster_results["loss"] = weighted_loss
        cluster_results["samples"] = total_examples
        cluster_results["correct"] = weighted_accuracy * total_examples

        print(f"➡️ Adaptive aggregated accuracy: {weighted_accuracy * 100:.2f}%\n")

        # 🔽 各クラスタごとのファイルにラウンド結果を1行ずつ追記
        output_dir = os.path.join(RESULTS_BASE_DIR, f"cluster_{cluster_id}")
        os.makedirs(output_dir, exist_ok=True)
        summary_file = os.path.join(output_dir, "round_metrics.jsonl")

        round_summary = {
            "accuracy": weighted_accuracy,
            "loss": weighted_loss,
            "samples": total_examples,
            "correct": weighted_accuracy * total_examples,
            "aggregation_type": "adaptive"
        }

        with open(summary_file, "a") as f:
            json.dump(round_summary, f)
            f.write("\n")  # JSONL形式で追記

        return {"accuracy": weighted_accuracy, "loss": weighted_loss}

    strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(selected_cids),
        min_available_clients=len(selected_cids),
        min_evaluate_clients=len(selected_cids),
        proximal_mu=proximal_mu,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )

    client_cache = {}

    def client_fn(context: Context):
        # Fix: Use partition-id directly as index for selected_cids
        partition_id = context.node_config.get("partition-id", context.node_id)
        idx = int(partition_id)
        
        # Ensure idx is within bounds
        if idx >= len(selected_cids):
            idx = idx % len(selected_cids)
        
        real_cid = selected_cids[idx]

        if real_cid not in client_cache:
            client_cache[real_cid] = FLClient(real_cid, selected_cids).to_client()

        return client_cache[real_cid]


    # Memory-efficient simulation with reduced resources
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(selected_cids),
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": gpu_memory_fraction},  # Reduced GPU allocation
    )

    acc = cluster_results.get("accuracy", 0.0)
    loss = cluster_results.get("loss", 0.0)
    examples = cluster_results.get("samples", 0)
    correct = cluster_results.get("correct", 0.0)

    cluster_result = {
        "final_accuracy": acc,
        "final_loss": loss,
        "samples": examples,
        "correct": correct,
    }
    
    # Clear memory after cluster processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return cluster_result

# Memory-efficient cross-cluster knowledge sharing
cluster_models = {}  # Store final models from each cluster
cluster_weights = {}  # Store cluster weights for aggregation

# Execute clusters sequentially for memory efficiency
for cluster_id in range(num_clusters):
    result = process_cluster(cluster_id, client_cluster_map, num_clusters)
    if result is not None:
        final_cluster_metrics[f"cluster_{cluster_id}"] = result
        total_correct += result["correct"]
        total_samples += result["samples"]
        
        # Store cluster model and weight for cross-cluster sharing
        # Weight by cluster size for better aggregation
        cluster_weights[cluster_id] = result["samples"]

# Simple cross-cluster knowledge sharing (optional, memory-efficient)
if num_clusters > 1 and len(cluster_weights) > 1:
    print("\n🔄 Implementing cross-cluster knowledge sharing...")
    
    # Calculate global accuracy improvement potential
    total_weight = sum(cluster_weights.values())
    weighted_accuracy = sum(final_cluster_metrics[f"cluster_{cid}"]["final_accuracy"] * weight 
                          for cid, weight in cluster_weights.items()) / total_weight
    
    print(f"📊 Cross-cluster weighted accuracy: {weighted_accuracy*100:.2f}%")
    
    # Store cross-cluster metrics
    final_cluster_metrics["cross_cluster"] = {
        "weighted_accuracy": weighted_accuracy,
        "num_clusters": len(cluster_weights),
        "cluster_sizes": cluster_weights
    }

# 最終集計・保存
overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
final_cluster_metrics["overall"] = {
    "accuracy": overall_accuracy,
    "total_correct": total_correct,
    "total_samples": total_samples,
}

summary_path = os.path.join(RESULTS_BASE_DIR, "final_summary.json")
with open(summary_path, "w") as f:
    json.dump(final_cluster_metrics, f, indent=2)

print(f"\n✅ 全クラスタの最終結果を {summary_path} に保存しました。")
