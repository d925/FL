# server.py
import flwr as fl
from flwr.server import ServerConfig
from config import num_rounds,num_clients
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

def fit_config_fn(rnd: int):
    return {"proximal_mu": 0.01}

# FedProx 戦略の設定
strategy = fl.server.strategy.FedProx(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=aggregate_metrics,
    on_fit_config_fn=fit_config_fn,
    min_fit_clients=10,         # 学習を開始するために最低10クライアントの参加を要求
    min_available_clients=num_clients,   # 少なくとも10クライアントが利用可能であることを要求
    proximal_mu=0.001,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )