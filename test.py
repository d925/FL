import os
import json
from config import num_clients, num_rounds, is_cluster
from utils import generate_and_save_dirichlet_partitioned_data, get_partitioned_data, num_labels
from cluster_test import cluster_clients_kmeans_dual, cluster_clients_with_metadata
from model import CNN
import torch
import torch.optim as optim
import torch.nn as nn

# 結果保存ディレクトリ作成
RESULTS_BASE_DIR = "results"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

generate_and_save_dirichlet_partitioned_data(num_clients)

if is_cluster:
    # Step 2: クラスタリング実行
    #cluster_clients_kmeans_dual(num_clients=num_clients)
    cluster_clients_with_metadata(num_clients=num_clients)
else:
    client_cluster_map = {cid: 0 for cid in range(num_clients)}  # 全クライアントをクラスタ0に所属させる
    cluster_list = [0]
    num_clusters = 1

#PS C:\Users\sharu\Documents\FL_project\FL> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#PS C:\Users\sharu\Documents\FL_project\FL> & "C:\Users\sharu\Miniconda3\shell\condabin\conda-hook.ps1"
#PS C:\Users\sharu\Documents\FL_project\FL> conda activate fl_cpu_env