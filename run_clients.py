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

# Step 3: 各クラスタに対してフェデレーテッドラーニングを実行
for cluster_id in range(num_clusters):
    print(f"\n=== Starting Federated Learning for Cluster {cluster_id} ===\n")

    # 現在のクラスタに属するクライアントIDを取得
    client_ids = [cid for cid, c in client_cluster_map.items() if c == cluster_id]
    num_cluster_clients = len(client_ids)

    # サーバープロセスを起動（1クラスタに対して1回）
    server_env = os.environ.copy()
    server_env.update({
        "CLUSTER_ID": str(cluster_id),
        "NUM_CLUSTER_CLIENTS": str(num_cluster_clients),
        "TOTAL_NUM_CLIENTS": str(num_clients)
    })
    server_proc = subprocess.Popen(["python", "server.py"], env=server_env)
    time.sleep(5)  # サーバー起動待機（必要に応じて調整）

    # クライアントプロセス群を起動
    client_procs = []
    for client_id in client_ids:
        gpu_id = client_id % torch.cuda.device_count()
        client_env = os.environ.copy()
        client_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        proc = subprocess.Popen(["python", "client.py", str(client_id)], env=client_env)
        client_procs.append(proc)

    # クライアントすべての終了を待機
    for proc in client_procs:
        proc.wait()

    # サーバープロセスの終了を待機
    server_proc.wait()
