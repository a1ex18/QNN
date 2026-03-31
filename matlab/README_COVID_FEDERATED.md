# COVID-19 Federated Learning with Over-the-Air Aggregation

Last updated: 2026-03-30

This folder contains COVID-19 X-ray federated learning experiments with wireless channel simulation.

## Overview

The code simulates federated learning where model updates from multiple users (hospitals/sites) are transmitted over a noisy MIMO wireless channel instead of perfect digital communication, studying the impact of communication errors on model convergence.

## Files

### Python Training Entry Points
- `main_covid_federated.py` - Main COVID-19 federated learning script (recommended)
- `main_24_12_4user.py` - MATLAB-style Python port now configured for COVID-19 dataset

### COVID-19 Components
- `covid_data_loader.py` - Loads and partitions COVID-19 Radiography Dataset across federated users
- `covid_cnn_model.py` - Lightweight CNN model and train/eval helpers
- `requirements_covid_federated.txt` - Python dependencies for COVID-19 implementation

### Shared Utilities
- `channel_utils.py` - MIMO channel simulation (AWGN, quantization, wireless transmission)

### Legacy MATLAB Reference
- `main_24_12_4user.m` - Original MATLAB script
- `cnn_train1.m` - MATLAB training helper
- `requirements_main_24_12_4user.txt` - Legacy dependency list

## COVID-19 Dataset

### Dataset Structure
The COVID-19 Radiography Dataset should be located at:
```
/home/kali/Desktop/QNN/COVID-19_Radiography_Dataset/
├── COVID/
│   ├── images/     (3,616 images)
│   └── masks/
├── Lung_Opacity/
│   ├── images/     (6,012 images)
│   └── masks/
├── Normal/
│   ├── images/     (10,192 images)
│   └── masks/
└── Viral Pneumonia/
    ├── images/     (1,345 images)
    └── masks/
```

### Classification Tasks
- **Binary** (faster): COVID vs Normal (2 classes)
- **Multiclass**: COVID, Lung_Opacity, Normal, Viral Pneumonia (4 classes)

### Default Dataset Path
Python scripts are configured to use:
`/home/kali/Desktop/QNN/COVID-19_Radiography_Dataset`

## Installation

```bash
source /home/kali/Desktop/QNN/.venv/bin/activate
cd /home/kali/Desktop/QNN/matlab
pip install -r requirements_covid_federated.txt
```

## Usage

### Run Recommended COVID-19 Federated Learning

```bash
source /home/kali/Desktop/QNN/.venv/bin/activate
cd /home/kali/Desktop/QNN/matlab
python main_covid_federated.py
```

### Run MATLAB-Style COVID Port

```bash
source /home/kali/Desktop/QNN/.venv/bin/activate
cd /home/kali/Desktop/QNN/matlab
python main_24_12_4user.py
```

**Expected Output:**
1. Loads COVID-19 X-ray images (this takes ~1-2 minutes for full dataset)
2. Partitions data across 4 users/hospitals
3. Trains local CNN models per user
4. Runs federated rounds with wireless channel simulation:
   - Aggregates user models
   - Transmits over noisy MIMO channel (SNR = -4 dB)
   - Each user receives corrupted weights
   - Local retraining continues
5. Plots accuracy curves and BER (Bit Error Rate)
6. Saves:
   - `covid_federated_results.png` - Accuracy and BER plots
   - `covid_global_model.pth` - Final trained model

### Configuration (main_covid_federated.py)

Edit `main_covid_federated.py` to customize:

```python
# Classification task
num_classes = 2          # 2 for binary, 4 for multiclass

# Training parameters
local_epochs = 2         # Epochs per user per round
num_rounds = 20          # Total federated rounds
batch_size = 16          # Mini-batch size
lr = 0.0001             # Learning rate

# Wireless channel
SNRdB = -4              # Signal-to-noise ratio in dB
Nt, Nr = 4, 32          # MIMO antenna configuration
rep = 3                 # Repetition coding factor
```

## Model Architecture

### COVID-19 CNN
```
Conv2d(3 → 32) → BatchNorm → ReLU → MaxPool(2)     # 224×224 → 112×112
Conv2d(32 → 64) → BatchNorm → ReLU → MaxPool(2)    # 112×112 → 56×56
Conv2d(64 → 128) → BatchNorm → ReLU → MaxPool(2)   # 56×56 → 28×28
GlobalAvgPool → FC(128 → 64) → Dropout(0.5) → FC(64 → classes)
```

**Parameters:** ~177K trainable parameters (manageable for wireless transmission)

## Wireless Channel Simulation

The Python implementations use the same channel model:

1. **Quantization**: Float32 weights → 8-bit integers
2. **Encoding**: Bits → bipolar symbols {-1, +1}
3. **Repetition**: Each bit repeated `rep` times for reliability
4. **MIMO + AWGN**: 4×32 antenna configuration with additive Gaussian noise
5. **Decoding**: Soft decision (averaging) + thresholding
6. **Dequantization**: Reconstruct float32 weights

**BER Tracking**: Bit Error Rate is computed and logged for each transmission.

## Performance Notes

### Speed Optimization
- **Use binary classification** (2 classes) for faster experimentation
- **Reduce image size**: Change `img_size=(128, 128)` in `main_covid_federated.py`
- **Fewer users**: 2 users instead of 4
- **Fewer rounds**: Start with 5-10 rounds for testing

### Memory Optimization
- Lower `batch_size` if running out of GPU/RAM
- Use CPU mode if GPU memory insufficient: edit device selection in main script

### Expected Runtime (full binary dataset, 4 users, 20 rounds)
- CPU: ~30-60 minutes
- GPU: ~10-20 minutes

## Research Questions

This code enables investigation of:
1. How wireless channel noise affects federated convergence
2. Trade-offs between SNR and model accuracy
3. Impact of repetition coding on reliability vs efficiency
4. Comparison: medical imaging vs sensor data under channel noise
5. Heterogeneous data distributions across hospitals

## Troubleshooting

### Import Errors
```bash
# If torchvision missing
pip install torchvision --no-cache-dir
```

### Out of Memory
- Reduce `batch_size` to 8 or 4
- Use binary classification (2 classes)
- Reduce `img_size` to (128, 128)

### Slow Loading
- First run loads all images into memory (~1-2 minutes)
- Subsequent rounds are fast
- Consider using a smaller subset for testing

### Dataset Not Found
- Ensure COVID-19_Radiography_Dataset is at: `/home/kali/Desktop/QNN/COVID-19_Radiography_Dataset/`
- Or modify `DATASET_ROOT`/`dataset_root` in:
   - `main_24_12_4user.py`
   - `covid_data_loader.py`
   - `main_covid_federated.py`

## Citation

If using this code, please cite:
- COVID-19 Radiography Dataset: [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

## License

Research/Educational use. Check individual dataset licenses.

## Dependencies

- See `requirements_covid_federated.txt` for the recommended Python stack.
- For legacy script compatibility, use `requirements_main_24_12_4user.txt`.
