import flwr as fl
from flwr.server import ServerConfig
from config import num_rounds
import json
import os

RESULTS_PATH = "round_metrics.json"
current_round = 0 

def save_metrics_to_json(round_number, accuracy, loss):
    data = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            data = json.load(f)
    
    data[f"round_{round_number}"] = {"accuracy": accuracy, "loss": loss}
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)

def aggregate_metrics(results):
    global current_round
    print(f"[Server] Raw results from clients:\n{results}\n")

    total_examples = sum(num_examples for num_examples, _ in results)

    accuracies = [metrics["accuracy"] * num_examples for num_examples, metrics in results]
    losses = [metrics["loss"] * num_examples for num_examples, metrics in results]

    avg_accuracy = sum(accuracies) / total_examples
    avg_loss = sum(losses) / total_examples
    current_round += 1

    print(f"\n[Server] Round summary → Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}\n")
    save_metrics_to_json(current_round, avg_accuracy, avg_loss)

    return {"accuracy": avg_accuracy, "loss": avg_loss}

# Flower strategy + evaluate_metrics_aggregation_fn に渡す
strategy = fl.server.strategy.FedProx(
    fraction_fit=0.5,
    fraction_evaluate=1.0,
    #evaluate_metrics_aggregation_fn=aggregate_metrics, 
    proximal_mu=0.01,
)

fl.server.start_server(
    server_address="localhost:8080",
    config=ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)
