import flwr as fl
from typing import Dict
import tensorflow as tf
from model_def import create_cnn_model
from data_loader import build_federated_datasets

# Build datasets once; each client will pick its shard

_client_datasets, val_ds, test_ds = build_federated_datasets()

class FederatedClient(fl.client.NumPyClient):
    # Method: __init__ - Helper routine for   init   logic.
    # Parameters: `self` is class instance reference; `cid` is cid input value.
    def __init__(self, cid: str):
        self.cid = cid
        self.model = create_cnn_model()
        self.train_ds = _client_datasets[cid]
        self.val_ds = val_ds

    # Method: get_parameters - Helper routine for get parameters logic.
    # Parameters: `self` is class instance reference; `config` is config input value.
    def get_parameters(self, config: Dict):
        return self.model.get_weights()


    # Method: fit - Helper routine for fit logic.
    # Parameters: `self` is class instance reference; `parameters` is parameters input value; `config` is config input value.
    def fit(self, parameters, config):
        import numpy as np
        import os
        self.model.set_weights(parameters)
        local_epochs = int(config.get("local_epochs", 1))
        weights_dir = f"client_weights_{self.cid}"
        os.makedirs(weights_dir, exist_ok=True)
        for epoch in range(local_epochs):
            self.model.fit(self.train_ds, epochs=1, verbose=0)
            # Save weights and biases for each layer after this epoch
            weights = self.model.get_weights()
            np.save(os.path.join(weights_dir, f"weights_epoch_{epoch+1}.npy"), weights, allow_pickle=True)
            print(f"[Client {self.cid}] Epoch {epoch+1} weights and biases saved.")
            # Print weights and biases summary and full arrays for this epoch
            for i, w in enumerate(weights):
                print(f"[Client {self.cid}] Layer {i} weights/biases (epoch {epoch+1}): shape={w.shape}, mean={np.mean(w):.4f}, std={np.std(w):.4f}")
                print(f"[Client {self.cid}] Layer {i} weights/biases array (epoch {epoch+1}):\n{w}\n")
                # Print biases if this is a 1D array (typical for biases)
                if len(w.shape) == 1:
                    print(f"[Client {self.cid}] Layer {i} BIASES (epoch {epoch+1}): {w}\n")
        # Final weights and biases to be sent to server
        final_weights = self.model.get_weights()
        print(f"[Client {self.cid}] FINAL weights and biases to be sent to server:")
        for i, w in enumerate(final_weights):
            print(f"[Client {self.cid}] Layer {i} weights/biases: shape={w.shape}, mean={np.mean(w):.4f}, std={np.std(w):.4f}")
            print(f"[Client {self.cid}] Layer {i} FINAL weights/biases array:\n{w}\n")
            # Print biases if this is a 1D array (typical for biases)
            if len(w.shape) == 1:
                print(f"[Client {self.cid}] Layer {i} FINAL BIASES: {w}\n")
        # Save the final weights and biases sent to the server
        np.save(os.path.join(weights_dir, f"final_weights_sent_to_server.npy"), final_weights, allow_pickle=True)
        print(f"[Client {self.cid}] Final weights and biases sent to server saved at {os.path.join(weights_dir, 'final_weights_sent_to_server.npy')}")
        return final_weights, len(list(self.train_ds.unbatch())), {}

    # Method: evaluate - Helper routine for evaluate logic.
    # Parameters: `self` is class instance reference; `parameters` is parameters input value; `config` is config input value.
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.val_ds, verbose=0)
        return float(loss), len(list(self.val_ds.unbatch())), {"accuracy": float(acc)}


# Function: client_fn - Helper routine for client fn logic.
# Parameters: `cid` is cid input value.
def client_fn(cid: str):
    return FederatedClient(cid)



# Function: main - Helper routine for main logic.
# Parameters: none.
def main():
    import os
    cid = os.environ.get("CLIENT_ID", "client_1")
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=FederatedClient(cid))

if __name__ == "__main__":
    main()
