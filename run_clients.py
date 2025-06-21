import subprocess
import torch
import os
import time
from config import num_clients, num_clusters
from utils import generate_and_save_dirichlet_partitioned_data
from cluster import cluster_clients

# Step 1: パーティション作成
generate_and_save_dirichlet_partitioned_data(num_clients)

# Step 2: クラスタリング
client_cluster_map = cluster_clients(num_clients=num_clients, num_clusters=num_clusters)

# Step 3: 各クラスタごとにFL実行
for cluster_id in range(num_clusters):
    print(f"\n=== Starting Federated Learning for Cluster {cluster_id} ===\n")
    
    # 対象クライアントを抽出
    client_ids = [cid for cid, c in client_cluster_map.items() if c == cluster_id]
    num_cluster_clients = len(client_ids)
    
    # server.py を subprocess で起動
    server_env = dict(os.environ, 
                      CLUSTER_ID=str(cluster_id), 
                      NUM_CLUSTER_CLIENTS=str(num_cluster_clients),
                      TOTAL_NUM_CLIENTS=str(num_clients))  # 使いたければ
    server_proc = subprocess.Popen(["python", "server.py"], env=server_env)
    time.sleep(5)  # サーバー起動待機

    # 各クライアントを起動
    client_procs = []
    for client_id in client_ids:
        gpu_id = client_id % torch.cuda.device_count()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id), CLIENT_ID=str(client_id))
        proc = subprocess.Popen(["python", "client.py", str(client_id)], env=env)
        client_procs.append(proc)

    # クライアント完了を待機
    for proc in client_procs:
        proc.wait()

    # サーバー終了待機
    server_proc.wait()

