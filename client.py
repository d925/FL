import sys
import json
import flwr as fl
import torch
import torch.optim as optim
from model import MobileNetV2_FL
from utils import get_partitioned_data
from config import num_clients

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
        global_params = [p.clone().detach() for p in self.model.parameters()]  # ← グローバルパラメータ保存

        self.model.train()
        mu = config.get("proximal_mu", 0.01)  # ← サーバから渡されるかデフォルト

        for _ in range(3):
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Proximal term: μ/2 * ||w - w_global||²
                prox_term = 0.0
                for param, global_param in zip(self.model.parameters(), global_params):
                    prox_term += ((param - global_param.to(self.device)) ** 2).sum()
                loss += (mu / 2) * prox_term

                loss.backward()
                self.optimizer.step()
        self.log("Finished local training with FedProx")
        return self.get_parameters(config), len(self.trainloader.dataset), {}
    def fit_config(rnd):
        return {"proximal_mu": 0.01}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        avg_loss = total_loss / len(self.testloader.dataset)
        accuracy = correct / len(self.testloader.dataset)
        self.log(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
        return avg_loss, len(self.testloader.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    client_id = int(sys.argv[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(LABEL_ASSIGN_PATH, "r") as f:
        label_info = json.load(f)
        num_labels = label_info["num_total_labels"]  # 👈 ここ！
    model = MobileNetV2_FL(num_classes=num_labels).to(device)

    trainset, testset = get_partitioned_data(client_id, num_clients)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FLClient(model, trainloader, testloader, client_id=client_id, device=device)
    )
