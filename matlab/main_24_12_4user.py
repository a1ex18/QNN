"""
Python port of main_24_12_4user.m: federated learning with over-the-air
aggregation over a noisy MIMO channel.
Trains on COVID-19 Radiography Dataset.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from PIL import Image

from channel_utils import (
    build_data_to_signal,
    spat_mod_wt,
    signals_wt,
    beaming_bias,
)

# ---------------------------------------------------------------------------
# COVID-19 Data Loading
# ---------------------------------------------------------------------------

DATASET_ROOT = '/home/kali/Desktop/QNN/COVID-19_Radiography_Dataset'

def load_covid_images(dataset_root=None, img_size=(128, 128), num_classes=2):
    """Load COVID-19 Radiography Dataset images and labels."""
    if dataset_root is None:
        dataset_root = DATASET_ROOT
    from torchvision import transforms
    
    dataset_root = Path(dataset_root)
    
    if num_classes == 2:
        class_folders = ['COVID', 'Normal']
    elif num_classes == 4:
        class_folders = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    else:
        raise ValueError("num_classes must be 2 or 4")
    
    images_list = []
    labels_list = []
    
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    
    print(f"Loading {num_classes}-class COVID-19 dataset from {dataset_root}...")
    
    for class_idx, class_name in enumerate(class_folders):
        class_path = dataset_root / class_name / 'images'
        if not class_path.exists():
            print(f"Warning: {class_path} not found, skipping...")
            continue
        
        image_files = list(class_path.glob('*.png'))[:500]  # Limit for faster loading
        print(f"  {class_name}: {len(image_files)} images")
        
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img)
                img_array = img_tensor.permute(1, 2, 0).numpy()
                images_list.append(img_array)
                labels_list.append(class_idx)
            except Exception as e:
                continue
    
    images = np.stack(images_list, axis=0).astype(np.float32)
    labels = np.array(labels_list, dtype=np.int64)
    
    print(f"Loaded {len(images)} images with shape {images.shape}")
    return images, labels, class_folders


# Function: partition_covid_data - Helper routine for partition covid data logic.
# Parameters: `images` is images input value; `labels` is labels input value; `num_users` is num users input value; `test_fraction` is test fraction input value; `seed` is seed input value.
def partition_covid_data(images, labels, num_users=4, test_fraction=0.2, seed=42):
    """Partition COVID data across users for federated learning."""
    rng = np.random.default_rng(seed)
    
    indices = np.arange(len(images))
    rng.shuffle(indices)
    images = images[indices]
    labels = labels[indices]
    
    n_test = int(len(images) * test_fraction)
    XTest = images[:n_test]
    YTest = labels[:n_test]
    XTrain_all = images[n_test:]
    YTrain_all = labels[n_test:]
    
    XTrain = [[] for _ in range(num_users)]
    YTrain = [[] for _ in range(num_users)]
    
    for i, (img, lbl) in enumerate(zip(XTrain_all, YTrain_all)):
        user_idx = i % num_users
        XTrain[user_idx].append(img)
        YTrain[user_idx].append(lbl)
    
    XTrain = [np.stack(user_data, axis=0) for user_data in XTrain]
    YTrain = [np.array(user_labels, dtype=np.int64) for user_labels in YTrain]
    
    print(f"\nData partitioning:")
    print(f"  Test set: {len(XTest)} samples")
    for i in range(num_users):
        print(f"  User {i+1}: {len(XTrain[i])} samples")
    
    return XTrain, YTrain, XTest, YTest


def load_covid_federated(dataset_root=None, num_users=4, img_size=(128, 128), 
                         num_classes=2, test_fraction=0.2):
    """Load and partition COVID-19 data in one call."""
    if dataset_root is None:
        dataset_root = DATASET_ROOT
    
    images, labels, class_names = load_covid_images(
        dataset_root, img_size=img_size, num_classes=num_classes
    )
    
    XTrain, YTrain, XTest, YTest = partition_covid_data(
        images, labels, num_users=num_users, test_fraction=test_fraction
    )
    
    return XTrain, YTrain, [XTest], [YTest], class_names

# ---------------------------------------------------------------------------
# Model: CNN for COVID-19
# ---------------------------------------------------------------------------

class CovidCNN(nn.Module):
    """Lightweight CNN for COVID-19 X-ray classification."""
    # Method: __init__ - Helper routine for   init   logic.
    # Parameters: `self` is class instance reference; `num_classes` is num classes input value; `input_channels` is input channels input value.
    def __init__(self, num_classes=2, input_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    # Method: forward - Helper routine for forward logic.
    # Parameters: `self` is class instance reference; `x` is input value for computation.
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    # Method: get_flat_weights - Helper routine for get flat weights logic.
    # Parameters: `self` is class instance reference.
    def get_flat_weights(self):
        weights = []
        for param in self.parameters():
            weights.append(param.detach().cpu().numpy().ravel())
        return np.concatenate(weights)
    
    # Method: set_flat_weights - Helper routine for set flat weights logic.
    # Parameters: `self` is class instance reference; `flat_weights` is flat weights input value.
    def set_flat_weights(self, flat_weights):
        offset = 0
        with torch.no_grad():
            for param in self.parameters():
                num_params = param.numel()
                param_data = flat_weights[offset:offset+num_params]
                param.copy_(torch.tensor(param_data.reshape(param.shape), dtype=torch.float32))
                offset += num_params


# Function: cnn_train1 - Helper routine for cnn train1 logic.
# Parameters: `x_train` is x train input value; `y_train` is y train input value; `model` is model instance used for training/inference; `options` is options input value; `device` is device input value; `num_classes` is num classes input value.
def cnn_train1(x_train, y_train, model, options, device, num_classes):
    """
    Train CNN model and return weights.
    model: a CovidCNN instance.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=options["InitialLearnRate"])
    
    # Images: (N, H, W, C) -> (N, C, H, W)
    x = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    y = torch.tensor(y_train, dtype=torch.long, device=device)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=options["MiniBatchSize"], shuffle=True, drop_last=False)
    
    for _ in range(options["MaxEpochs"]):
        for xb, yb in dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
    
    flat_w = model.get_flat_weights()
    # Split into 4 parts for compatibility with existing code structure
    quarter = len(flat_w) // 4
    w1 = flat_w[:quarter]
    w2 = flat_w[quarter:2*quarter]
    b1 = flat_w[2*quarter:3*quarter]
    b2 = flat_w[3*quarter:]
    return w1, w2, b1, b2, model


# Function: classify - Helper routine for classify logic.
# Parameters: `net` is net input value; `x` is input value for computation; `device` is device input value.
def classify(net, x, device):
    """Classify and return predicted class indices."""
    net.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        logits = net(x_t)
        pred = logits.argmax(dim=1).cpu().numpy()
    return pred


# Function: reconstruct_cnn_weights - Helper routine for reconstruct cnn weights logic.
# Parameters: `w1` is w1 input value; `w2` is w2 input value; `b1` is b1 input value; `b2` is b2 input value.
def reconstruct_cnn_weights(w1, w2, b1, b2):
    """Reconstruct flat weights from 4 parts for CNN."""
    return np.concatenate([w1.ravel(), w2.ravel(), b1.ravel(), b2.ravel()])


# Function: main_covid - Helper routine for main covid logic.
# Parameters: none.
def main_covid():
    """Run federated learning with COVID-19 dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("\n" + "="*60)
    print("COVID-19 X-RAY FEDERATED LEARNING")
    print("="*60 + "\n")

    # Load COVID-19 dataset
    dataset_root = DATASET_ROOT
    num_classes = 2  # Binary classification
    num_users = 4
    
    XTrain, YTrain, XTest, YTest, class_names = load_covid_federated(
        dataset_root=dataset_root,
        num_users=num_users,
        img_size=(128, 128),
        num_classes=num_classes,
        test_fraction=0.2
    )
    
    print(f"\nClass names: {class_names}")
    
    # Calculate normalization factors
    total_dataset = sum(len(XTrain[i]) for i in range(num_users))
    norms = [len(XTrain[i]) / total_dataset for i in range(num_users)]
    
    # Training hyperparameters
    local_epochs = 2
    num_rounds = 10
    batch_size = 16
    lr = 0.0001
    options = {
        "MaxEpochs": local_epochs,
        "MiniBatchSize": batch_size,
        "InitialLearnRate": lr,
        "Shuffle": "every-epoch",
        "Verbose": 0,
    }
    
    # Wireless channel parameters
    SNRdB = -4
    dist = 1
    Nt, Nr = 4, 32
    rep = 3
    
    # Create models for each user
    print(f"\nInitializing {num_users} CNN models...")
    models = [CovidCNN(num_classes=num_classes).to(device) for _ in range(num_users)]
    
    # Initial training per user
    print(f"\nInitial per-user training ({local_epochs} epochs)...")
    user_weights = []
    for i in range(num_users):
        print(f"  User {i+1}:")
        w1, w2, b1, b2, models[i] = cnn_train1(
            XTrain[i], YTrain[i], models[i], options, device, num_classes
        )
        user_weights.append((w1, w2, b1, b2))
    
    # Evaluate initial performance
    XTest1, YTest1 = XTest[0], YTest[0]
    print("\nInitial per-user test accuracy:")
    for i in range(num_users):
        acc = np.mean(classify(models[i], XTest1, device) == YTest1)
        print(f"  User {i+1}: {acc:.4f}")
    
    # Tracking arrays
    accg = np.zeros(num_rounds)
    dev_accs = [np.zeros(num_rounds) for _ in range(num_users)]
    BER_weights = [np.zeros(num_rounds) for _ in range(num_users)]
    
    print(f"\n{'='*60}")
    print(f"Federated rounds: {num_rounds}, SNR: {SNRdB} dB")
    print(f"{'='*60}\n")
    
    t_start = time.perf_counter()
    
    for round_idx in range(num_rounds):
        print(f"Round {round_idx + 1}/{num_rounds}")
        
        w1_1, w2_1, b1_1, b2_1 = user_weights[0]
        w1_2, w2_2, b2_1_u2, b2_2 = user_weights[1]
        w1_3, w2_3, b1_3, b2_3 = user_weights[2]
        w1_4, w2_4, b1_4, b2_4 = user_weights[3]
        
        # Aggregate: weighted average
        w1 = sum(norms[i] * user_weights[i][0] for i in range(num_users))
        w2 = sum(norms[i] * user_weights[i][1] for i in range(num_users))
        b1 = sum(norms[i] * user_weights[i][2] for i in range(num_users))
        b2 = sum(norms[i] * user_weights[i][3] for i in range(num_users))
        
        # Create global model
        global_model = CovidCNN(num_classes=num_classes).to(device)
        global_flat = reconstruct_cnn_weights(w1, w2, b1, b2)
        global_model.set_flat_weights(global_flat)
        
        # Fine-tune global on test set
        wg_1, wg_2, bg_1, bg_2, global_model = cnn_train1(
            XTest1, YTest1, global_model, options, device, num_classes
        )
        global_flat = reconstruct_cnn_weights(wg_1, wg_2, bg_1, bg_2)
        
        accg[round_idx] = np.mean(classify(global_model, XTest1, device) == YTest1)
        print(f"  Global accuracy: {accg[round_idx]:.4f}")
        
        # Transmit global weights to users via wireless channel
        transmitted_weights = []
        for i in range(num_users):
            w_signal = build_data_to_signal(global_flat)
            decoded_w, BER_weights[i][round_idx] = spat_mod_wt(Nt, Nr, SNRdB, w_signal, dist, rep)
            
            if len(decoded_w) < len(global_flat):
                decoded_w = np.pad(decoded_w, (0, len(global_flat) - len(decoded_w)))
            else:
                decoded_w = decoded_w[:len(global_flat)]
            
            transmitted_weights.append(decoded_w)
        
        print(f"  BER: {[f'{BER_weights[i][round_idx]:.6f}' for i in range(num_users)]}")
        
        # Local retraining
        for i in range(num_users):
            models[i].set_flat_weights(transmitted_weights[i])
            w1, w2, b1, b2, models[i] = cnn_train1(
                XTrain[i], YTrain[i], models[i], options, device, num_classes
            )
            user_weights[i] = (w1, w2, b1, b2)
            dev_accs[i][round_idx] = np.mean(classify(models[i], XTrain[i], device) == YTrain[i])
        
        print(f"  Local train accs: {[f'{dev_accs[i][round_idx]:.4f}' for i in range(num_users)]}\n")
    
    elapsed = time.perf_counter() - t_start
    print(f"{'='*60}")
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"{'='*60}\n")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(accg, 'k-', linewidth=2, label="Global")
    for i in range(num_users):
        plt.plot(dev_accs[i], '--', label=f"User {i+1}")
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"COVID-19 Federated Learning (SNR={SNRdB}dB)")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for i in range(num_users):
        plt.plot(BER_weights[i], label=f"User {i+1}")
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("BER")
    plt.title("Wireless Channel BER")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("covid_federated_results.png", dpi=150)
    plt.show()
    print("Plot saved to covid_federated_results.png")


if __name__ == "__main__":
    main_covid()
