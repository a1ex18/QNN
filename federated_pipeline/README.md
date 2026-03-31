# Federated Learning Pipeline (COVID-19 Radiography)

Last updated: 2026-03-30

This component trains a COVID-19 X-ray classifier using Flower-based federated learning.

## Files

- `data_loader.py`: Dataset loading, partitioning, and `tf.data` pipelines.
- `model_def.py`: Shared MobileNetV2-based model definition.
- `server.py`: Flower server with FedProx strategy.
- `client.py`: Flower NumPy client.
- `run_federated.py`: Single-process sequential simulation (no networking).
- `evaluation.py`: Post-training metrics and confusion matrix generation.

## Dataset

Expected dataset root is configured in `data_loader.py` via `DATASET_ROOT`.
Current default targets the COVID-19 Radiography dataset folders (`COVID`, `Normal`, etc.).

## Quick Start

Run sequential simulation (fast local sanity check):

```bash
python federated_pipeline/run_federated.py
```

Evaluate saved model:

```bash
python federated_pipeline/evaluation.py
```

## Distributed Flower Mode

Start in separate terminals:

```bash
# Terminal 1
python federated_pipeline/server.py

# Terminal 2
CLIENT_ID=client_1 python federated_pipeline/client.py

# Terminal 3
CLIENT_ID=client_2 python federated_pipeline/client.py
```

## Configuration Notes

- `NUM_CLIENTS`, batch size, and split settings: `data_loader.py`
- Round and local training settings: `server.py` / `run_federated.py`
- Keep server and simulation round counts aligned when comparing results.

## Dependencies

Use root `requirements.txt` for this pipeline.

```bash
pip install -r requirements.txt
```
