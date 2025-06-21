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

print("クラスタリング結果:")
for client_id, cluster_id in client_cluster_map.items():
    print(f"クライアント {client_id} は クラスター {cluster_id}")
# Step 3: 各クラスタごとにフェデレーテッドラーニングを実行
server_proc = subprocess.Popen(["python", "server.py"])
time.sleep(10)  # サーバーの起動を待つ

# 全クライアントを起動
client_procs = []
for client_id in range(num_clients):
    gpu_id = client_id % torch.cuda.device_count()  # 複数GPUがある場合、分散して割り当て
    client_env = os.environ.copy()
    client_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    proc = subprocess.Popen(["python", "client.py", str(client_id)], env=client_env)
    client_procs.append(proc)

# クライアント全ての終了を待つ
for proc in client_procs:
    proc.wait()

# サーバーの終了を待つ
server_proc.wait()