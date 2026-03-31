import flwr as fl
from typing import Dict
from model_def import create_cnn_model
from data_loader import build_federated_datasets
import tensorflow as tf

# Global validation dataset for server evaluation

_client_datasets, val_ds, test_ds = build_federated_datasets()

# Build a fresh global model
GLOBAL_MODEL = create_cnn_model()


# Function: get_evaluate_fn - Helper routine for get evaluate fn logic.
# Parameters: `model` is model instance used for training/inference.
def get_evaluate_fn(model):
    # Returns a function for server-side evaluation
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict):
        model.set_weights(parameters)
        loss, acc = model.evaluate(val_ds, verbose=0)
        return float(loss), {"accuracy": float(acc)}
    return evaluate


# Function: main - Helper routine for main logic.
# Parameters: none.
def main():
    strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(_client_datasets),
        min_evaluate_clients=len(_client_datasets),
        min_available_clients=len(_client_datasets),
        on_evaluate_config_fn=lambda rnd: {},
        proximal_mu=0.1,
        evaluate_fn=get_evaluate_fn(GLOBAL_MODEL),
        initial_parameters=fl.common.ndarrays_to_parameters(GLOBAL_MODEL.get_weights()),
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )

if __name__ == "__main__":
    main()
