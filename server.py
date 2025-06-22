# server.py
import flwr as fl
from flwr.server import ServerConfig
from config import num_rounds,num_clients
import json
import os
import sys

cluster_id = int(sys.argv[1])
RESULTS_DIR = f"results/cluster_{cluster_id}"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(RESULTS_DIR, "round_metrics.json")
current_round = 0

def save_metrics_to_json(round_number, accuracy, loss, total_examples):
    data = {}
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r") as f:
                data = json.load(f)
        except Exception as e:
            print("Error reading metrics file:", e)
    data[f"round_{round_number}"] = {
        "accuracy": accuracy,
        "loss": loss,
        "total_examples": total_examples
    }
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
    save_metrics_to_json(current_round, avg_accuracy, avg_loss, total_examples)
    return {"accuracy": avg_accuracy, "loss": avg_loss}
# FedProx 戦略の設定
strategy = fl.server.strategy.FedProx(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=aggregate_metrics,
    proximal_mu=0,
    min_fit_clients=1,
    min_available_clients=1,
    min_evaluate_clients=1,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )