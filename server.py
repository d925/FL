# server.py
import flwr as fl
from flwr.server import ServerConfig
import torch
from model import CNN
from config import num_rounds,num_clients
from utils import get_shared_dataset_loader,prepare_shared_dataset
import json
import os

RESULTS_PATH = "round_metrics.json"
current_round = 0

def save_metrics_to_json(round_number, accuracy, loss):
    data = {}
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r") as f:
                data = json.load(f)
        except Exception as e:
            print("Error reading metrics file:", e)
    data[f"round_{round_number}"] = {"accuracy": accuracy, "loss": loss}
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)

def aggregate_metrics(results):
    global current_round
    print(f"[Server] Raw results from clients:\n{results}\n")
    
    # クライアントから返ってくる (num_examples, metrics) の形式を想定
    total_examples = sum(num_examples for num_examples, _ in results if num_examples > 0)
    total_examples = total_examples if total_examples > 0 else 1
    accuracies = [metrics["accuracy"] * num_examples for num_examples, metrics in results if num_examples > 0]
    losses = [metrics["loss"] * num_examples for num_examples, metrics in results if num_examples > 0]
    
    avg_accuracy = sum(accuracies) / total_examples
    avg_loss = sum(losses) / total_examples
    current_round += 1

    print(f"\n[Server] Round summary → Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}\n")
    save_metrics_to_json(current_round, avg_accuracy, avg_loss)
    return {"accuracy": avg_accuracy, "loss": avg_loss}
# FedProx 戦略の設定
strategy = fl.server.strategy.FedProx(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=aggregate_metrics,
    min_fit_clients=int(num_clients / 2),         # 学習を開始するために最低10クライアントの参加を要求
    min_available_clients=num_clients,   # 少なくとも10クライアントが利用可能であることを要求
    proximal_mu=0.01,
)

def pretrain_on_shared_dataset():
    print("[Server] Pretraining on shared dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN.get_model().to(device)
    model.train()
    loader = get_shared_dataset_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print("[Server] Pretraining complete.")
    return model.state_dict()


if __name__ == "__main__":
    prepare_shared_dataset()
    initial_parameters = pretrain_on_shared_dataset()

    strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        min_fit_clients=int(num_clients / 2),
        min_available_clients=num_clients,
        proximal_mu=0,
        initial_parameters=fl.common.ndarrays_to_parameters([
            t.detach().cpu().numpy() for t in initial_parameters.values()
        ])
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
