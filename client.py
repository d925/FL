import sys
import json
import flwr as fl
import torch
import torch.optim as optim
from model import CNN
from utils import get_partitioned_data,num_labels
from config import num_clients
import os

LABEL_ASSIGN_PATH = "label_assignments.json"

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, client_id=0, device="cpu"):
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.client_id = client_id
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def log(self, msg):
        colors = ["\033[94m", "\033[92m", "\033[93m", "\033[95m", "\033[91m"]
        reset = "\033[0m"
        color = colors[self.client_id % len(colors)]
        print(f"{color}[Client {self.client_id}] {msg}{reset}")

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, val in zip(self.model.parameters(), parameters):
            p.data = torch.from_numpy(val).to(self.device).to(torch.float32)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_params = [p.clone().detach() for p in self.model.parameters()]

        self.model.train()
        mu = config.get("proximal_mu", 0.01)

        for _ in range(3):
            for data, target in self.trainloader:
                print("learning")
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                prox_term = 0.0
                for param, global_param in zip(self.model.parameters(), global_params):
                    prox_term += ((param - global_param.to(self.device)) ** 2).sum()
                loss += (mu / 2) * prox_term
                loss.backward()
                self.optimizer.step()
        self.log("Finished local training with FedProx")
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # evaluate関数内の最初の方に追加
        with torch.no_grad():
            all_labels = []
            for _, target in self.testloader:
                all_labels += target.tolist()
            max_label = max(all_labels)
            min_label = min(all_labels)
            print(f"[DEBUG] Evaluationラベル範囲: {min_label}〜{max_label}")
            assert max_label < self.model.fc2.out_features, f"💥 評価ラベル {max_label} が num_classes を超えてる"
        print("f")
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        avg_loss = total_loss / len(self.testloader.dataset)
        accuracy = correct / len(self.testloader.dataset)
        self.log(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        return avg_loss, len(self.testloader.dataset), {"accuracy": accuracy, "loss": avg_loss}

if __name__ == "__main__":
    client_id = int(sys.argv[1])
    """
    # --- ここからGPU初期化追加 ---
    # 親プロセスから渡されたCUDA_VISIBLE_DEVICESに合わせてGPU固定
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices:
        devices = visible_devices.split(",")
        assigned_gpu = int(devices[0])  # 先頭のGPUを使う想定
    else:
        assigned_gpu = 0

    torch.cuda.set_device(assigned_gpu)

    # ここでプロセスのGPUメモリ使用量制限をかける（任意、メモリ足りないなら調整）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.5, device=assigned_gpu)  # 50%に制限例

    # --- ここまでGPU初期化追加 ---
    """
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=num_labels).to(device)
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, 64, 64))

    trainset, testset = get_partitioned_data(client_id, num_clients)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FLClient(model, trainloader, testloader, client_id=client_id, device=device)
    )
