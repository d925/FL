import sys
import json
import flwr as fl
import torch
import torch.optim as optim
from model import MobileNetV2_FL
from utils import get_partitioned_data  # 以前の data_utils.py と同等の内容を想定
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
        # 適宜学習率やオプティマイザは調整可能
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def log(self, msg):
        # クライアントごとに色分けして出力
        colors = ["\033[94m", "\033[92m", "\033[93m", "\033[95m", "\033[91m"]
        reset = "\033[0m"
        color = colors[self.client_id % len(colors)]
        print(f"{color}[Client {self.client_id}] {msg}{reset}")

    def get_parameters(self, config):
        # モデルパラメータを NumPy 配列に変換して返す
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        # 受け取った NumPy 配列を元にモデルパラメータを更新する
        for p, val in zip(self.model.parameters(), parameters):
            p.data = torch.from_numpy(val).to(self.device).to(torch.float32)

    def fit(self, parameters, config):
        # サーバからグローバルパラメータを受け取って学習開始
        self.set_parameters(parameters)
        # ローカル更新の前にグローバルパラメータを保存（FedProx 用）
        global_params = [p.clone().detach() for p in self.model.parameters()]

        self.model.train()
        # サーバ側で設定された proximal_mu を取得（なければ 0.01）
        mu = config.get("proximal_mu", 0.01)

        # エポック回数等は必要に応じてループ回数を増やす
        for _ in range(1):
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # FedProx のための proximal term を計算する
                prox_term = 0.0
                for param, global_param in zip(self.model.parameters(), global_params):
                    prox_term += ((param - global_param.to(self.device)) ** 2).sum()
                loss += (mu / 2) * prox_term

                loss.backward()
                self.optimizer.step()
        self.log("Finished local training with FedProx")
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # 評価時はまずサーバからのパラメータで更新する
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

    # ※ fit_config はサーバ側で用いられる設定関数のため、クライアントクラス内で定義する必要はない
    # ここでは不要のためコメントアウト
    # def fit_config(rnd):
    #     return {"proximal_mu": 0.01}


if __name__ == "__main__":
    # クライアントID をコマンドライン引数から取得
    client_id = int(sys.argv[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ラベル割当ファイルから総ラベル数を取得
    try:
        with open(LABEL_ASSIGN_PATH, "r") as f:
            label_info = json.load(f)
        num_labels = label_info["num_total_labels"]
    except Exception as e:
        print("Error reading label assignments:", e)
        sys.exit(1)

    # モデルの初期化（取得したラベル総数を出力層の次元として利用）
    model = MobileNetV2_FL(num_classes=num_labels).to(device)

    # 各クライアント用にパーティション化済みデータを取得
    trainset, testset = get_partitioned_data(client_id, num_clients)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=0
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=128)

    # Flower のクライアントとしてサーバに参加
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FLClient(model, trainloader, testloader, client_id=client_id, device=device),
    )