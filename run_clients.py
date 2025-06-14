# main.py
import subprocess
from config import num_clients
from utils import generate_and_save_dirichlet_partitioned_data
import torch
import os

generate_and_save_dirichlet_partitioned_data(num_clients)

processes = []
for client_id in range(num_clients):
    gpu_id = client_id % torch.cuda.device_count()  # 実GPU番号
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))  # ←これは optional（制限）
    proc = subprocess.Popen(["python", "client.py", str(client_id), str(gpu_id)], env=env)
    processes.append(proc)
    
for proc in processes:
    proc.wait()
