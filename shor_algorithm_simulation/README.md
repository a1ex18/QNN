# Shor Algorithm Simulation (Qiskit)

Last updated: 2026-03-30

Fork source: https://github.com/ttlion/ShorAlgQiskit

This directory contains multiple Shor variants with different qubit/resource trade-offs.

## Recommended Environment

- Python 3.10+
- `qiskit==0.23.1`
- `qiskit-aer==0.12.0`

Optional GPU backend:

```bash
pip install qiskit-aer-gpu-cu11
```

## Run Variants

```bash
python3 shor_algorithm_simulation/Shor_Sequential_QFT.py
python3 shor_algorithm_simulation/Shor_Sequential_AQFT.py
python3 shor_algorithm_simulation/Shor_Sequential_QFT_gpu.py
```

## Tests

```bash
python3 shor_algorithm_simulation/Test_QFT.py
python3 shor_algorithm_simulation/Test_Mult.py
python3 shor_algorithm_simulation/Test_classical_before_quantum.py
python3 shor_algorithm_simulation/Test_classical_after_quantum.py
```

## Notes

- Sequential variants reduce qubit usage compared to normal QFT mode.
- Some scripts use older Qiskit API symbols; keep versions pinned for compatibility.
