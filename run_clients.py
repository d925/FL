# main.py
import subprocess
from config import num_clients
from utils import generate_and_save_dirichlet_partitioned_data, prepare_label_indices
import torch
import os

generate_and_save_dirichlet_partitioned_data(num_clients)
prepare_label_indices()

processes = []
for client_id in range(num_clients):
    gpu_id = client_id % torch.cuda.device_count()  # GPU数に応じて割り振り
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
    proc = subprocess.Popen(["python", "client.py", str(client_id)], env=env)
    processes.append(proc)

for proc in processes:
    proc.wait()
