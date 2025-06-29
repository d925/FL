import json
import os
import flwr as fl
import torch
import torch.optim as optim
from flwr.server import ServerConfig
from config import num_clients, num_clusters, num_rounds
from utils import generate_and_save_dirichlet_partitioned_data, get_partitioned_data, num_labels
from cluster import cluster_clients
from model import CNN

RESULTS_BASE_DIR = "results"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

# データ生成とクラスタリング
generate_and_save_dirichlet_partitioned_data(num_clients)
client_cluster_map = cluster_clients(num_clients=num_clients, num_clusters=num_clusters)

print("クラスタリング結果:")
for cid, clust_id in client_cluster_map.items():
    print(f"クライアント {cid} は クラスター {clust_id}")

# FLClient定義
class FLClient(fl.client.NumPyClient):
    def __init__(self, cid, selected_cids):
        self.cid = int(cid)
        self.selected_cids = selected_cids
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN(num_classes=num_labels).to(self.device)
        self.model.eval()
        with torch.no_grad():
            self.model(torch.randn(1, 3, 64, 64).to(self.device))

        if self.cid in selected_cids:
            trainset, testset = get_partitioned_data(self.cid, num_clients)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=32)
        else:
            self.trainloader = None
            self.testloader = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, val in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(val, dtype=torch.float32, device=self.device)

    def fit(self, parameters, config):
        if self.trainloader is None:
            return self.get_parameters(config), 0, {}

        self.set_parameters(parameters)
        mu = config.get("proximal_mu", 0.0)
        global_params = [p.detach().clone() for p in self.model.parameters()]
        self.model.train()

        for _ in range(5):
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                if mu > 0:
                    prox = sum(((param - gparam.to(self.device)) ** 2).sum() for param, gparam in zip(self.model.parameters(), global_params))
                    loss += (mu / 2) * prox
                loss.backward()
                self.optimizer.step()

        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        if self.testloader is None:
            return 0.0, 0, {"accuracy": 0.0, "loss": 0.0}

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
        avg_loss = total_loss / len(self.testloader.dataset)
        accuracy = correct / len(self.testloader.dataset)
        return avg_loss, len(self.testloader.dataset), {"accuracy": accuracy, "loss": avg_loss}

# 各クラスタを順に実行
final_cluster_metrics = {}
total_correct = 0
total_samples = 0

for cluster_id in range(num_clusters):
    selected_cids = [cid for cid, cl in client_cluster_map.items() if cl == cluster_id]
    print(f"\n--- クラスタ {cluster_id} のシミュレーション開始 ---")

    def client_fn(context):
        return FLClient(context.client_id, selected_cids).to_client()

    # 戦略
    def aggregate_metrics(results):
        total = sum(num for num, _ in results)
        if total == 0:
            return {"accuracy": 0.0, "loss": 0.0}
        accuracy = sum(metrics["accuracy"] * num for num, metrics in results) / total
        loss = sum(metrics["loss"] * num for num, metrics in results) / total
        return {"accuracy": accuracy, "loss": loss}

    strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(selected_cids),
        min_evaluate_clients=len(selected_cids),
        min_available_clients=len(selected_cids),
        proximal_mu=0.0,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )

    # シミュレーション実行
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # 精度記録
    accs = [r.metrics["accuracy"] for r in history.global_evaluation if "accuracy" in r.metrics]
    losses = [r.metrics["loss"] for r in history.global_evaluation if "loss" in r.metrics]
    examples = [r.num_examples for r in history.global_evaluation]

    if accs:
        final_accuracy = accs[-1]
        final_loss = losses[-1]
        total = examples[-1]
        final_cluster_metrics[f"cluster_{cluster_id}"] = {
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
            "samples": total,
            "correct": final_accuracy * total,
        }
        total_correct += final_accuracy * total
        total_samples += total

# 集計・保存
overall_accuracy = total_correct / total_samples if total_samples else 0.0
final_cluster_metrics["overall"] = {
    "accuracy": overall_accuracy,
    "total_correct": total_correct,
    "total_samples": total_samples,
}
summary_path = os.path.join(RESULTS_BASE_DIR, "final_summary.json")
with open(summary_path, "w") as f:
    json.dump(final_cluster_metrics, f, indent=2)

print(f"\n✅ 全クラスタの最終結果を {summary_path} に保存しました。")
