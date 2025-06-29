import json
import os
from config import num_clients, num_clusters, num_rounds
from utils import generate_and_save_dirichlet_partitioned_data
from cluster import cluster_clients
import flwr as fl
from flwr.server import ServerConfig

RESULTS_BASE_DIR = "results"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

# Step 1: Dirichlet 分割のデータ作成
generate_and_save_dirichlet_partitioned_data(num_clients)

# Step 2: クラスタリング
client_cluster_map = cluster_clients(num_clients=num_clients, num_clusters=num_clusters)

print("クラスタリング結果:")
for cid, clust_id in client_cluster_map.items():
    print(f"クライアント {cid} は クラスター {clust_id}")

final_cluster_metrics = {}
total_correct = 0
total_samples = 0

# 共通のclient_fnを定義
from model import CNN
from utils import get_partitioned_data, num_labels
import torch
import torch.optim as optim

class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id, selected_clients):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN(num_classes=num_labels).to(self.device)
        self.model.eval()
        with torch.no_grad():
            self.model(torch.randn(1, 3, 64, 64).to(self.device))
        
        # クラスタに含まれるクライアントだけロード
        if self.client_id in selected_clients:
            trainset, testset = get_partitioned_data(self.client_id, num_clients)
        else:
            # クライアントがこのクラスタに含まれなければ空のデータセットなど（適宜調整）
            trainset, testset = [], []
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True) if trainset else []
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=32) if testset else []
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, val in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(val, dtype=torch.float32, device=self.device)

    def fit(self, parameters, config):
        if not self.trainloader:
            return self.get_parameters(config), 0, {}
        self.set_parameters(parameters)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
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
                    prox = sum(((param - gparam.to(self.device))**2).sum() for param, gparam in zip(self.model.parameters(), global_params))
                    loss += (mu/2) * prox
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        if not self.testloader:
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

def client_fn(cid, selected_clients):
    return FLClient(int(cid), selected_clients)

# Step 3: クラスタごとのFL実行
for cluster_id in range(num_clusters):
    cluster_clients_list = [cid for cid, cl in client_cluster_map.items() if cl == cluster_id]

    print(f"\n--- クラスタ {cluster_id} のシミュレーション開始 ---")

    strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(cluster_clients_list),
        min_evaluate_clients=len(cluster_clients_list),
        min_available_clients=len(cluster_clients_list),
        proximal_mu=0.0,
    )

    # クラスタのクライアントのみを動かすclient_fnラッパー
    def client_fn_wrapper(cid):
        return client_fn(cid, cluster_clients_list)

    fl.simulation.start_simulation(
        client_fn=client_fn_wrapper,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # 結果読み込み
    metrics_path = os.path.join(RESULTS_BASE_DIR, f"cluster_{cluster_id}", "round_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            round_data = json.load(f)
            if round_data:
                last_key = sorted(round_data.keys(), key=lambda x: int(x.split("_")[1]))[-1]
                acc = round_data[last_key]["accuracy"]
                loss = round_data[last_key]["loss"]
                total_examples = round_data[last_key]["total_examples"]
                correct = acc * total_examples
                total_correct += correct
                total_samples += total_examples
                final_cluster_metrics[f"cluster_{cluster_id}"] = {
                    "final_accuracy": acc,
                    "final_loss": loss,
                    "samples": total_examples,
                    "correct": correct
                }

# 全体精度の集計と保存
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
