# main.py
import subprocess
from config import num_clients
from utils import generate_label_assignments, prepare_label_indices

# ラベル割当とデータインデックスの準備
generate_label_assignments(num_clients)
prepare_label_indices()

processes = []
for client_id in range(num_clients):
    proc = subprocess.Popen(["python", "client.py", str(client_id)])
    processes.append(proc)

for proc in processes:
    proc.wait()