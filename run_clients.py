import os
import json
from config import num_clients, num_clusters, num_rounds
from utils import generate_and_save_dirichlet_partitioned_data, get_partitioned_data, num_labels
from cluster import cluster_clients
from model import CNN
import flwr as fl
from flwr.server import ServerConfig
import torch
import torch.optim as optim
import torch.nn as nn
from flwr.client import NumPyClient

# 結果保存ディレクトリ作成
RESULTS_BASE_DIR = "results"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

# Step 1: Dirichlet分割生成
generate_and_save_dirichlet_partitioned_data(num_clients)

# Step 2: クラスタリング実行
client_cluster_map = cluster_clients(num_clients=num_clients, num_clusters=num_clusters)

print("クラスタリング結果:")
for cid, clust_id in client_cluster_map.items():
    print(f"クライアント {cid} は クラスター {clust_id}")

# クラスタ単位の最終結果保持用
final_cluster_metrics = {}
total_correct = 0
total_samples = 0

# クライアントクラス定義
class FLClient(NumPyClient):
    def __init__(self, cid, active_cids):
        self.cid = int(cid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN(num_classes=num_labels).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        if self.cid in active_cids:
            trainset, testset = get_partitioned_data(self.cid, num_clients)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=32)
        else:
            self.trainloader = []
            self.testloader = []

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, val in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(val, dtype=torch.float32, device=self.device)

    def fit(self, parameters, config):
        if not self.trainloader:
            return self.get_parameters(config), 0, {}
        self.set_parameters(parameters)
        global_params = [p.clone().detach() for p in self.model.parameters()]
        mu = config.get("proximal_mu", 0.0)

        self.model.train()
        for _ in range(5):
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

# Step 3: クラスタごとのFL実行ループ
for cluster_id in range(1):
    selected_cids = [cid for cid, clid in client_cluster_map.items() ]
    print(f"\n--- クラスタ {cluster_id} のシミュレーション開始 ---")

    cluster_results = {}

    def aggregate_metrics(results):
        print("\n📊 このラウンドのクライアント評価結果:")
        for i, (num_examples, metrics) in enumerate(results):
            acc = metrics["accuracy"] * 100
            loss = metrics["loss"]
            print(f"  Client {i}: Accuracy = {acc:.2f}%, Loss = {loss:.4f}, Samples = {num_examples}")

        total_examples = sum(num_examples for num_examples, _ in results if num_examples > 0)
        total_examples = total_examples if total_examples > 0 else 1
        weighted_accuracy = sum(metrics["accuracy"] * num_examples for num_examples, metrics in results if num_examples > 0)
        weighted_loss = sum(metrics["loss"] * num_examples for num_examples, metrics in results if num_examples > 0)
        avg_accuracy = weighted_accuracy / total_examples
        avg_loss = weighted_loss / total_examples

        cluster_results["accuracy"] = avg_accuracy
        cluster_results["loss"] = avg_loss
        cluster_results["samples"] = total_examples
        cluster_results["correct"] = avg_accuracy * total_examples

        return {"accuracy": avg_accuracy, "loss": avg_loss}

    strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(selected_cids),
        min_available_clients=len(selected_cids),
        min_evaluate_clients=len(selected_cids),
        proximal_mu=0.0,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )

    def client_fn(cid: str):
        idx = int(cid)  # Flowerから渡されるクラスタ内連番cidは文字列
        real_cid = selected_cids[idx]
        return FLClient(real_cid, selected_cids).to_client()



    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(selected_cids),
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1.0},
    )

    acc = cluster_results.get("accuracy", 0.0)
    loss = cluster_results.get("loss", 0.0)
    examples = cluster_results.get("samples", 0)
    correct = cluster_results.get("correct", 0.0)

    final_cluster_metrics[f"cluster_{cluster_id}"] = {
        "final_accuracy": acc,
        "final_loss": loss,
        "samples": examples,
        "correct": correct,
    }

    total_correct += correct
    total_samples += examples

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
