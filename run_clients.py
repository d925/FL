# run_clients_fedprox.py  （あなたの既存スクリプトをそのまま置き換える想定）

import os
import json
import random
import shutil
import numpy as np
import torch
from config import num_clients, num_rounds, is_cluster
from utils import generate_and_save_dirichlet_partitioned_data, get_partitioned_data, num_labels
from cluster import cluster_clients
from cluster_test import cluster_clients_kmeans_dual, cluster_clients_with_metadata
from cluster_wasserstein import cluster_clients_with_metadata_ratio
from model import CNN
import flwr as fl
from flwr.server import ServerConfig
import torch.optim as optim
import torch.nn as nn
from flwr.client import NumPyClient

# -------------------
# FedProx ハイパーパラメータ
# 0 にすると標準の FedAvg と等価
FEDPROX_MU = 0.1
# -------------------

# 乱数シード完全固定
SEED = 44
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 結果保存ディレクトリ作成（前回削除）
RESULTS_BASE_DIR = "results"
if os.path.exists(RESULTS_BASE_DIR):
    shutil.rmtree(RESULTS_BASE_DIR)
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

generate_and_save_dirichlet_partitioned_data(num_clients)



if is_cluster:
    #client_cluster_map = cluster_clients(num_clients=num_clients)
    client_cluster_map = cluster_clients_with_metadata_ratio(num_clients=num_clients)
    print("クラスタリング結果:")
    for cid, clust_id in client_cluster_map.items():
        print(f"クライアント {cid} は クラスター {clust_id}")
    cluster_list = sorted(set(client_cluster_map.values()))
    num_clusters = len(cluster_list)
else:
    client_cluster_map = {cid: 0 for cid in range(num_clients)}
    cluster_list = [0]
    num_clusters = 1

final_cluster_metrics = {}
total_correct = 0
total_samples = 0

# -------------------------
# FLClient（FedProx対応）
# -------------------------
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
        # グローバルパラメータをモデルにセット
        self.set_parameters(parameters)

        # FedProx: "global_params" を保存（勾配計算対象外）
        global_params = [p.detach().clone() for p in self.model.parameters()]

        # オプティマイザ再作成（必要ならハイパラをここで変える）
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.8)

        self.model.train()
        prev_loss = float('inf')
        EPOCHS = 1  # 現状 1 エポック（必要なら増やす）
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # ---------- FedProx の proximal term を追加 ----------
                if self.mu > 0:
                    prox_term = 0.0
                    for p, g in zip(self.model.parameters(), global_params):
                        prox_term += torch.sum((p - g.to(self.device)) ** 2)
                    # 正規化（サンプル数やパラメータ数で割る場合はここで調整）
                    loss = loss + (self.mu / 2.0) * prox_term
                # ----------------------------------------------------

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            scheduler.step()

            # Early stop-like 挙動（変化が小さければ break）
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
        avg_loss = total_loss / len(self.testloader.dataset)
        accuracy = correct / len(self.testloader.dataset)
        self.log(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        return avg_loss, len(self.testloader.dataset), {"accuracy": accuracy, "loss": avg_loss}


# Step 3: クラスタごとのFL実行ループ（以降は従来コードと同様）
for cluster_id in range(num_clusters):
    selected_cids = [cid for cid, clid in client_cluster_map.items() if clid == cluster_id]
    print(f"\n--- クラスタ {cluster_id} のシミュレーション開始 ---")

    cluster_results = {}
    round_counter = 0  # aggregate_metrics の外にグローバル変数で追加


    
    def aggregate_metrics(results):
        global round_counter
        round_counter += 1  # ← ラウンド番号カウント

        print(f"\n📊 第 {round_counter} ラウンドのクライアント評価結果:")

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
            print(f"  Client {i}: Acc={acc*100:.2f}%, Loss={loss:.4f}, Samples={num_examples}")

        avg_accuracy = weighted_sum_acc / total_weight
        avg_loss = weighted_sum_loss / total_weight

        cluster_results.setdefault("round_accuracies", []).append(avg_accuracy)
        cluster_results.setdefault("round_losses", []).append(avg_loss)

        cluster_results["accuracy"] = avg_accuracy
        cluster_results["loss"] = avg_loss
        cluster_results["samples"] = total_examples
        cluster_results["correct"] = avg_accuracy * total_examples

        print(f"➡️ 第 {round_counter} ラウンドの平均精度: {avg_accuracy * 100:.2f}%\n")

        # ファイル保存
        output_dir = os.path.join(RESULTS_BASE_DIR, f"cluster_{cluster_id}")
        os.makedirs(output_dir, exist_ok=True)
        summary_file = os.path.join(output_dir, "round_metrics.jsonl")

        round_summary = {
            "round": round_counter,
            "accuracy": avg_accuracy,
            "loss": avg_loss,
            "samples": total_examples,
            "correct": avg_accuracy * total_examples,
        }

        with open(summary_file, "a") as f:
            json.dump(round_summary, f)
            f.write("\n")

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
            # FedProx の μ をクライアントへ渡す（ここではグローバル定数を使用）
            client_cache[real_cid] = FLClient(real_cid, selected_cids, mu=FEDPROX_MU).to_client()
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

# 最終集計・保存（従来通り）
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
