import flwr as fl
from flwr.server import ServerConfig
from config import num_rounds

fl.server.start_server(
    server_address="localhost:8080",
    config=ServerConfig(num_rounds=num_rounds),
    strategy=fl.server.strategy.Fedprox(
        fraction_fit=0.5,
        proximal_mu=0.01,
    )
)
