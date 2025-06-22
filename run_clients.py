# run_all.py
import subprocess
import torch
import os
import time
import json
from config import num_clients, num_clusters
from utils import generate_and_save_dirichlet_partitioned_data
from cluster import cluster_clients

# Step 1: Dirichlet 分割のデータ作成
generate_and_save_dirichlet_partitioned_data(num_clients)

# Step 2: クラスタリング
client_cluster_map = cluster_clients(num_clients=num_clients, num_clusters=num_clusters)

print("クラスタリング結果:")
for client_id, cluster_id in client_cluster_map.items():
    print(f"クライアント {client_id} は クラスター {cluster_id}")

# 保存用構造
RESULTS_BASE_DIR = "results"
final_cluster_metrics = {}
total_correct = 0
total_samples = 0

# Step 3: クラスタごとの FL 実行
for cluster_id in range(num_clusters):
    cluster_clients_list = [cid for cid, c in client_cluster_map.items() if c == cluster_id]

    print(f"\n--- クラスタ {cluster_id} のサーバー起動 ---")
    server_proc = subprocess.Popen(["python", "server.py", str(cluster_id)])
    time.sleep(10)

    client_procs = []
    for client_id in cluster_clients_list:
        gpu_id = client_id % torch.cuda.device_count()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        proc = subprocess.Popen(["python", "client.py", str(client_id)], env=env)
        client_procs.append(proc)

    for proc in client_procs:
        proc.wait()
    server_proc.wait()

    # 各クラスタの最終精度を読み取る
    metrics_path = os.path.join(RESULTS_BASE_DIR, f"cluster_{cluster_id}", "round_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            round_data = json.load(f)
            if round_data:
                last_key = sorted(round_data.keys(), key=lambda x: int(x.split("_")[1]))[-1]
                acc = round_data[last_key]["accuracy"]
                loss = round_data[last_key]["loss"]
                cluster_sample_count = len(cluster_clients_list) * 1000  # 仮の総サンプル数
                correct = acc * cluster_sample_count
                total_correct += correct
                total_samples += cluster_sample_count
                final_cluster_metrics[f"cluster_{cluster_id}"] = {
                    "final_accuracy": acc,
                    "final_loss": loss,
                    "samples": cluster_sample_count,
                    "correct": correct
                }

# 全体精度の集計と保存
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
