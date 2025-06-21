# main.py
import subprocess
import torch
import os
import time
from config import num_clients
from utils import generate_and_save_dirichlet_partitioned_data

# データパーティションを生成
generate_and_save_dirichlet_partitioned_data(num_clients)

# サーバープロセスを起動
server_proc = subprocess.Popen(["python", "server.py"])
time.sleep(10)  # サーバー起動を待機

# クライアントプロセスを起動
client_procs = []
for client_id in range(num_clients):
    gpu_id = client_id % torch.cuda.device_count()
    client_env = os.environ.copy()
    client_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    proc = subprocess.Popen(["python", "client.py", str(client_id)], env=client_env)
    client_procs.append(proc)

# クライアントの終了を待機
for proc in client_procs:
    proc.wait()

# サーバーの終了を待機
server_proc.wait()
