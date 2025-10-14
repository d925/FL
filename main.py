import os
import json
import random
import numpy as np
import torch
from config import num_clients, num_rounds, is_cluster
from utils import generate_and_save_dirichlet_partitioned_data, get_partitioned_data, num_labels
from cluster_emb import cluster_clients_with_metadata_emb, get_client_metadata
from model_combined import CNNWithMetadata
import flwr as fl
from flwr.server import ServerConfig
import torch.optim as optim
import torch.nn as nn
from flwr.client import NumPyClient
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 乱数シード完全固定
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# ディレクトリ作成
# ============================================================
RESULTS_BASE_DIR = "results"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

# ============================================================
# データ生成（Dirichlet分割）
# ============================================================
generate_and_save_dirichlet_partitioned_data(num_clients)

# ============================================================
# メタデータエンコーダ初期化
# ============================================================
crops, diseases, regions = zip(*[get_client_metadata(cid) for cid in range(num_clients)])
crop_le, disease_le, region_le = LabelEncoder(), LabelEncoder(), LabelEncoder()
crop_le.fit(crops)
disease_le.fit(diseases)
region_le.fit(regions)

# ============================================================
# クラスタリング
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model import CNN  # 特徴抽出用ベースモデル
feature_extractor = CNN(num_classes=num_labels)
feature_extractor.fc2 = nn.Identity()
feature_extractor.to(device)

if is_cluster:
    client_cluster_map = cluster_clients_with_metadata_emb(
        model=feature_extractor,
        device=device,
        num_clients=num_clients
    )
    print("クラスタリング結果:")
    for cid, clust_id in enumerate(client_cluster_map):
        print(f"クライアント {cid} は クラスター {clust_id}")

    cluster_list = sorted(set(client_cluster_map))
    num_clusters = len(cluster_list)
else:
    client_cluster_map = [0 for _ in range(num_clients)]
    cluster_list = [0]
    num_clusters = 1

# ============================================================
# クラスタ単位の最終結果保持用
# ============================================================
final_cluster_metrics = {}
total_correct = 0
total_samples = 0

# ============================================================
# クライアント定義
# ============================================================
class FLClient(NumPyClient):
    def __init__(self, cid, active_cids):
        self.cid = int(cid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.cid in active_cids:
            trainset, testset = get_partitioned_data(self.cid, num_clients)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=32)
        else:
            self.trainloader, self.testloader = [], []

        # クライアントのcrop分布ベクトルを取得
        from cluster_emb import get_client_metadata_distribution
        crop_dist, _, _ = get_client_metadata_distribution(cid, num_clients, None)
        # crop分布ベクトルを固定順で作成
        all_crops = sorted(list(crop_le.classes_))
        self.crop_vec = torch.tensor([crop_dist.get(c, 0.0) for c in all_crops], dtype=torch.float32).to(self.device)

        self.model = CNNWithMetadata(
            num_classes=num_labels,
            num_crops=len(crop_le.classes_),
            num_diseases=len(disease_le.classes_),
            num_regions=len(region_le.classes_)
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def log(self, msg):
        print(f"[Client {self.cid}] {msg}")

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, val in zip(self.model.parameters(), parameters):
            p.data = torch.from_numpy(val).to(self.device).to(torch.float32)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.8)

        self.model.train()
        prev_loss = float('inf')
        for epoch in range(5):
            running_loss = 0.0
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                crop_vec_batch = self.crop_vec.unsqueeze(0).repeat(data.size(0), 1)
                output = self.model(data, crop_vec_batch)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            if abs(prev_loss - running_loss) < 1e-3:
                break
            prev_loss = running_loss

        self.log(f"Finished local training (final loss={prev_loss:.4f})")
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
                crop_vec_batch = self.crop_vec.unsqueeze(0).repeat(data.size(0), 1)
                output = self.model(data, crop_vec_batch)
                total_loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        avg_loss = total_loss / len(self.testloader.dataset)
        accuracy = correct / len(self.testloader.dataset)
        self.log(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        return avg_loss, len(self.testloader.dataset), {"accuracy": accuracy, "loss": avg_loss}

# ============================================================
# クラスタごとのFL実行
# ============================================================
for cluster_id in range(num_clusters):
    selected_cids = [cid for cid, clid in enumerate(client_cluster_map) if clid == cluster_id]
    print(f"\n--- クラスタ {cluster_id} のシミュレーション開始 ---")

    cluster_results = {}

    def aggregate_metrics(results):
        weighted_sum_acc, weighted_sum_loss, total_weight = 0.0, 0.0, 0.0
        total_examples = 0
        for i, (num_examples, metrics) in enumerate(results):
            acc = metrics["accuracy"]
            loss = metrics["loss"]
            weight = num_examples / (loss + 1e-6)
            weighted_sum_acc += acc * weight
            weighted_sum_loss += loss * weight
            total_weight += weight
            total_examples += num_examples
        avg_accuracy = weighted_sum_acc / total_weight
        avg_loss = weighted_sum_loss / total_weight

        cluster_results["accuracy"] = avg_accuracy
        cluster_results["loss"] = avg_loss
        cluster_results["samples"] = total_examples
        cluster_results["correct"] = avg_accuracy * total_examples
        return {"accuracy": avg_accuracy, "loss": avg_loss}

    strategy = fl.server.strategy.FedAvg(
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
            client_cache[real_cid] = FLClient(real_cid, selected_cids).to_client()
        return client_cache[real_cid]

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

# ============================================================
# 最終集計
# ============================================================
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
