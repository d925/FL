import subprocess
import torch
import os
import time
from config import num_clients, num_clusters
from utils import generate_and_save_dirichlet_partitioned_data
from cluster import cluster_clients

# Step 1: クライアントごとのデータパーティション作成（Dirichlet分布に基づく）
generate_and_save_dirichlet_partitioned_data(num_clients)

# Step 2: クラスタリングの実行
client_cluster_map = cluster_clients(num_clients=num_clients, num_clusters=num_clusters)

# Step 3: 各クラスタごとにフェデレーテッドラーニングを実行
for cluster_id in range(num_clusters):
    cluster_clients_list = [cid for cid, c in client_cluster_map.items() if c == cluster_id]

    print(f"--- クラスタ {cluster_id} のサーバー起動 ---")
    server_proc = subprocess.Popen(["python", "server.py", str(cluster_id)])
    time.sleep(5)  # サーバー起動を待機

    client_procs = []
    for client_id in cluster_clients_list:
        gpu_id = client_id % torch.cuda.device_count()
        client_env = os.environ.copy()
        client_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        proc = subprocess.Popen(["python", "client.py", str(client_id)], env=client_env)
        client_procs.append(proc)

    for proc in client_procs:
        proc.wait()

    server_proc.wait()
