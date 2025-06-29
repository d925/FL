# run_simulation.py
import os
import json
import torch
import flwr as fl
from flwr.server import ServerConfig
from flwr.common import Context

from config import num_clients, num_clusters, num_rounds
from utils import generate_and_save_dirichlet_partitioned_data, get_partitioned_data, num_labels
from cluster import cluster_clients
from model import CNN
import torch.nn as nn
import torch.optim as optim

RESULTS_BASE_DIR = "results"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

# ---------- Step 1: Dirichlet分割データの生成 ----------
generate_and_save_dirichlet_partitioned_data(num_clients)

# ---------- Step 2: クラスタリング ----------
client_cluster_map = cluster_clients(num_clients=num_clients, num_clusters=num_clusters)
print("クラスタリング結果:")
for cid, clid in client_cluster_map.items():
    print(f"クライアント {cid} は クラスター {clid}")

# ---------- クライアント定義 ----------
class FLClient(fl.client.NumPyClient):
    def __init__(self, cid: int):
        self.cid = int(cid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN(num_classes=num_labels).to(self.device)
        self.model.eval()
        with torch.no_grad():
            self.model(torch.randn(1, 3, 64, 64).to(self.device))

        trainset, testset = get_partitioned_data(self.cid, num_clients)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=32)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config): return [p.cpu().detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, val in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(val, dtype=torch.float32, device=self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        mu = config.get("proximal_mu", 0.0)
        global_params = [p.clone().detach() for p in self.model.parameters()]
        for _ in range(5):
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                if mu > 0:
                    prox = sum(((param - g.to(self.device)) ** 2).sum() for param, g in zip(self.model.parameters(), global_params))
                    loss += (mu / 2) * prox
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.criterion(output, target).item() * data.size(0)
                correct += (output.argmax(1) == target).sum().item()
        total = len(self.testloader.dataset)
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, total, {"accuracy": accuracy, "loss": avg_loss}

# ---------- シミュレーション ----------
final_cluster_metrics = {}
total_correct, total_samples = 0, 0

for cluster_id in range(num_clusters):
    cluster_cids = [str(cid) for cid, c in client_cluster_map.items() if c == cluster_id]
    print(f"\n--- クラスタ {cluster_id} のシミュレーション開始 ---")

    def client_fn(context: Context) -> fl.client.Client:
        cid = context.cid
        if cid in cluster_cids:
            return FLClient(cid).to_client()
        else:
            raise ValueError(f"Client {cid} not in cluster {cluster_id}")

    # 集約関数：精度と損失の重み付き平均を記録
    cluster_results = {}
    def aggregate_metrics(results):
        nonlocal cluster_results
        total = sum(num for num, _ in results)
        total = total if total > 0 else 1
        acc = sum(num * metrics["accuracy"] for num, metrics in results) / total
        loss = sum(num * metrics["loss"] for num, metrics in results) / total
        cluster_results = {"accuracy": acc, "loss": loss, "samples": total}
        print(f"[Cluster {cluster_id}] Accuracy: {acc:.4f}, Loss: {loss:.4f}")
        return {"accuracy": acc, "loss": loss}

    strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(cluster_cids),
        min_available_clients=len(cluster_cids),
        min_evaluate_clients=len(cluster_cids),
        proximal_mu=0.0,
        evaluate_metrics_aggregation_fn=aggregate_metrics
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(cluster_cids),
        client_resources={"num_cpus": 1},
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # 結果保存
    acc = cluster_results.get("accuracy", 0.0)
    loss = cluster_results.get("loss", 0.0)
    samples = cluster_results.get("samples", 0)
    correct = acc * samples
    total_correct += correct
    total_samples += samples
    final_cluster_metrics[f"cluster_{cluster_id}"] = {
        "final_accuracy": acc,
        "final_loss": loss,
        "samples": samples,
        "correct": correct,
    }

# ---------- 全体精度保存 ----------
overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
final_cluster_metrics["overall"] = {
    "accuracy": overall_accuracy,
    "total_correct": total_correct,
    "total_samples": total_samples
}

summary_path = os.path.join(RESULTS_BASE_DIR, "final_summary.json")
with open(summary_path, "w") as f:
    json.dump(final_cluster_metrics, f, indent=2)

print(f"\n✅ 全クラスタの最終結果を {summary_path} に保存しました。")
