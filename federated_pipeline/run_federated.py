"""Orchestrator script to launch a demo federated session sequentially.
For true parallelism, run server.py in one terminal and multiple instances of client.py (with different CLIENT_ID env vars) in others.
This script simulates sequential client updates per round for quick local testing.
"""

import flwr as fl
import os
from pathlib import Path
import re
from model_def import create_cnn_model
from data_loader import build_federated_datasets
from client import FederatedClient


ROUNDS = 8
LOCAL_EPOCHS = 3
NUM_CLIENTS = 8
BATCH_SIZE = 32

client_datasets, val_ds, test_ds = build_federated_datasets(num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE)

# Ensure checkpoint directory exists
ckpt_dir = Path("federated_pipeline/checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)


global_model = create_cnn_model()

# Resume support: find latest checkpoint and load
# Function: _latest_ckpt - Helper routine for  latest ckpt logic.
# Parameters: `ckpt_root` is ckpt root input value.
def _latest_ckpt(ckpt_root: Path):
    if not ckpt_root.exists():
        return None, 0
    round_files = list(ckpt_root.glob("global_round_*.h5"))
    last_round = 0
    last_path = None
    pattern = re.compile(r"global_round_(\d+)\.h5")
    for p in round_files:
        m = pattern.match(p.name)
        if m:
            r = int(m.group(1))
            if r > last_round:
                last_round = r
                last_path = p
    # If interrupt snapshot exists but no round ckpt, prefer interrupt
    intr = ckpt_root / "global_interrupt.h5"
    if intr.exists() and last_round == 0:
        return intr, 0
    return last_path, last_round

start_round = 1
ckpt_path, last_round = _latest_ckpt(ckpt_dir)
if ckpt_path is not None:
    try:
        global_model.load_weights(str(ckpt_path))
        start_round = last_round + 1 if last_round > 0 else 1
        print(f"Resumed weights from {ckpt_path}. Starting at round {start_round}.")
    except Exception as e:
        print(f"Warning: failed to load checkpoint {ckpt_path}: {e}")

try:
    for rnd in range(start_round, ROUNDS+1):
        print(f"\n=== Federated Round {rnd} ===")
        new_weights = []
        sizes = []
        for cid in client_datasets.keys():
            print(f"Client {cid} training...")
            c = FederatedClient(cid)
            c.model.set_weights(global_model.get_weights())
            # override dataset to avoid rebuilding
            c.train_ds = client_datasets[cid]
            history = c.model.fit(c.train_ds, epochs=LOCAL_EPOCHS, verbose=0)
            new_weights.append(c.model.get_weights())
            sizes.append(len(list(c.train_ds.unbatch())))
        # Aggregate (simple weighted average)
        total = sum(sizes)
        agg = []
        for layer_idx in range(len(new_weights[0])):
            layer_sum = sum(new_weights[i][layer_idx] * (sizes[i]/total) for i in range(len(new_weights)))
            agg.append(layer_sum)
        global_model.set_weights(agg)
        loss, acc = global_model.evaluate(val_ds, verbose=0)
        print(f"Round {rnd} validation - loss: {loss:.4f} acc: {acc:.4f}")
        # Save checkpoint after each round
        ckpt_path = ckpt_dir / f"global_round_{rnd}.h5"
        global_model.save(str(ckpt_path))
        print(f"Saved checkpoint: {ckpt_path}")
except KeyboardInterrupt:
    # Save an interrupt snapshot to avoid losing progress
    snap_path = ckpt_dir / "global_interrupt.h5"
    global_model.save(str(snap_path))
    print(f"\nTraining interrupted. Snapshot saved to {snap_path}")

print("\nEvaluating aggregated global model on test set...")
loss, acc = global_model.evaluate(test_ds, verbose=0)
print(f"Test loss: {loss:.4f} acc: {acc:.4f}")

global_model.save("federated_pipeline/global_model.h5")
print("Saved global model to federated_pipeline/global_model.h5")
