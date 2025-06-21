# server.py
import flwr as fl
from flwr.server import ServerConfig
from config import num_rounds, num_clients
import json
import os


# 結果保存用ディレクトリとファイルパス
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, f"round_metrics_cluster{cluster_id}.json")

current_round = 0

def save_metrics_to_json(round_number, accuracy, loss):
    data = {}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                data = json.load(f)
        except Exception as e:
            print("Error reading metrics file:", e)
    data[f"round_{round_number:02d}"] = {"accuracy": accuracy, "loss": loss}
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def aggregate_metrics(results):
    global current_round
    print(f"[Server] Raw results from clients:\n{results}\n")

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
    min_fit_clients=int(num_clients / 2),
    min_available_clients=num_clients,
    proximal_mu=0,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
