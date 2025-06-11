import flwr as fl
from flwr.server import ServerConfig
from config import num_rounds
import json
import os

RESULTS_PATH = "round_metrics.json"

def save_metrics_to_json(round_number, accuracy, loss):
    data = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            data = json.load(f)
    
    data[f"round_{round_number}"] = {"accuracy": accuracy, "loss": loss}
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)

def aggregate_metrics(results, server_round):
    accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
    losses = [r.loss * r.num_examples for _, r in results]
    num_examples = sum(r.num_examples for _, r in results)

    avg_accuracy = sum(accuracies) / num_examples
    avg_loss = sum(losses) / num_examples

    print(f"\n[Server] Round summary → Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}\n")
    # ✅ 正しい保存呼び出し：save_metrics_to_json()
    save_metrics_to_json(server_round, avg_accuracy, avg_loss)
    
    return {"accuracy": avg_accuracy, "loss": avg_loss}

# Flower strategy + evaluate_metrics_aggregation_fn に渡す
strategy = fl.server.strategy.FedProx(
    fraction_fit=0.2,
    fraction_evaluate=0.2,
    proximal_mu=0.01,
    evaluate_metrics_aggregation_fn=aggregate_metrics,
)

fl.server.start_server(
    server_address="localhost:8080",
    config=ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)
