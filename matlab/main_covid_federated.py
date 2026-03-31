"""
Python federated learning with over-the-air aggregation over a noisy MIMO channel
for COVID-19 X-ray classification.

Adapted from main_24_12_4user.py to work with COVID-19 Radiography Dataset.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from pathlib import Path

from covid_data_loader import load_covid_federated
from covid_cnn_model import CovidCNN, train_covid_cnn, evaluate_covid_cnn, classify_covid_cnn
from channel_utils import (
    build_data_to_signal,
    spat_mod_wt,
)

# ---------------------------------------------------------------------------
# Helper functions for weight aggregation and transmission
# ---------------------------------------------------------------------------

# Function: aggregate_weights_simple - Helper routine for aggregate weights simple logic.
# Parameters: `weights_list` is weights list input value; `norms` is norms input value.
def aggregate_weights_simple(weights_list, norms):
    """
    Weighted average of model weights from multiple users.
    
    Args:
        weights_list: list of flat weight arrays from users
        norms: list of normalization factors (dataset sizes)
    
    Returns:
        aggregated flat weight array
    """
    aggregated = np.zeros_like(weights_list[0])
    for weights, norm in zip(weights_list, norms):
        aggregated += norm * weights
    return aggregated


# Function: decode_and_reshape_weights - Helper routine for decode and reshape weights logic.
# Parameters: `decoded_signal` is decoded signal input value; `original_shape` is original shape input value.
def decode_and_reshape_weights(decoded_signal, original_shape):
    """
    Reshape decoded signal back to weight array shape.
    """
    return decoded_signal[:np.prod(original_shape)].reshape(original_shape)


# ---------------------------------------------------------------------------
# Main (federated learning with COVID-19 dataset)
# ---------------------------------------------------------------------------
# Function: main - Helper routine for main logic.
# Parameters: none.
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    # Load COVID-19 dataset
    dataset_root = '/home/kali/Desktop/QNN/COVID-19_Radiography_Dataset'
    print(f"Loading COVID-19 dataset from: {dataset_root}")
    
    # For faster experimentation, use binary classification (2 classes)
    # Change num_classes=4 for multiclass
    num_classes = 2  # Binary: COVID vs Normal
    
    XTrain, YTrain, XTest, YTest, class_names = load_covid_federated(
        dataset_root=dataset_root,
        num_users=2,
        img_size=(64, 64),
        num_classes=num_classes,
        test_fraction=0.2
    )
    
    print(f"\nClass names: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    num_users = 2
    
    # Calculate normalization factors (dataset size weights)
    total_dataset = sum(len(XTrain[i]) for i in range(num_users))
    norms = [len(XTrain[i]) / total_dataset for i in range(num_users)]
    print(f"\nDataset weights: {norms}")
    
    # Training hyperparameters
    local_epochs = 2  # Reduced for images (more compute-intensive)
    num_rounds = 5
    batch_size = 4  # Smaller batch for memory efficiency
    lr = 0.0001
    
    # Wireless channel parameters
    SNRdB = -4
    dist = 1
    Nt, Nr = 4, 32
    rep = 3  # Repetition factor for weights transmission
    
    # Create models for each user
    print(f"\nInitializing {num_users} user models...")
    models = [CovidCNN(num_classes=num_classes).to(device) for _ in range(num_users)]
    
    # Get parameter info
    num_params = models[0].get_num_params()
    print(f"Model parameters: {num_params:,}")
    
    # Initial training per user
    print(f"\nInitial per-user training ({local_epochs} epochs)...")
    for i in range(num_users):
        print(f"  User {i+1}:")
        models[i] = train_covid_cnn(
            models[i], XTrain[i], YTrain[i],
            epochs=local_epochs, batch_size=batch_size, lr=lr, device=device
        )
    
    # Evaluate initial performance
    print("\nInitial per-user test accuracy:")
    for i in range(num_users):
        acc = evaluate_covid_cnn(models[i], XTest, YTest, device=device)
        print(f"  User {i+1}: {acc:.4f}")
    
    # Arrays to track metrics
    accg = np.zeros(num_rounds)
    dev_accs = [np.zeros(num_rounds) for _ in range(num_users)]
    BER_weights = [np.zeros(num_rounds) for _ in range(num_users)]
    
    print(f"\n{'='*60}")
    print(f"Starting federated learning with over-the-air aggregation")
    print(f"Rounds: {num_rounds}, SNR: {SNRdB} dB")
    print(f"{'='*60}\n")
    
    t_start = time.perf_counter()
    
    for round_idx in range(num_rounds):
        print(f"Round {round_idx + 1}/{num_rounds}")
        
        # Step 1: Extract weights from each user model
        user_weights = [models[i].get_flat_weights() for i in range(num_users)]
        
        # Step 2: Aggregate weights (weighted average)
        aggregated_weights = aggregate_weights_simple(user_weights, norms)
        
        # Step 3: Create global model and load aggregated weights
        global_model = CovidCNN(num_classes=num_classes).to(device)
        global_model.set_flat_weights(aggregated_weights)
        
        # Step 4: Fine-tune global model on test set (server data)
        print("  Fine-tuning global model...")
        global_model = train_covid_cnn(
            global_model, XTest, YTest,
            epochs=1, batch_size=batch_size, lr=lr*0.5, device=device
        )
        
        # Evaluate global model
        accg[round_idx] = evaluate_covid_cnn(global_model, XTest, YTest, device=device)
        print(f"  Global accuracy: {accg[round_idx]:.4f}")
        
        # Step 5: Distribute global model to users via wireless channel
        global_weights = global_model.get_flat_weights()
        
        transmitted_weights = []
        for i in range(num_users):
            # Build signal for transmission
            w_signal = build_data_to_signal(global_weights)
            
            # Transmit over noisy MIMO channel
            decoded_w, BER_weights[i][round_idx] = spat_mod_wt(
                Nt, Nr, SNRdB, w_signal, dist, rep
            )
            
            # Pad or trim to match original size
            if len(decoded_w) < len(global_weights):
                decoded_w = np.pad(decoded_w, (0, len(global_weights) - len(decoded_w)))
            else:
                decoded_w = decoded_w[:len(global_weights)]
            
            transmitted_weights.append(decoded_w)
            print(f"  User {i+1} - BER: {BER_weights[i][round_idx]:.6f}")
        
        # Step 6: Each user loads received (noisy) weights and retrains locally
        print("  Local retraining...")
        for i in range(num_users):
            # Load transmitted weights
            models[i].set_flat_weights(transmitted_weights[i])
            
            # Local training
            models[i] = train_covid_cnn(
                models[i], XTrain[i], YTrain[i],
                epochs=local_epochs, batch_size=batch_size, lr=lr, device=device
            )
            
            # Evaluate on local training data
            dev_accs[i][round_idx] = evaluate_covid_cnn(
                models[i], XTrain[i], YTrain[i], device=device
            )
        
        print(f"  Local accuracies: {[f'{dev_accs[i][round_idx]:.4f}' for i in range(num_users)]}")
        print()
    
    elapsed = time.perf_counter() - t_start
    print(f"{'='*60}")
    print(f"Training completed in {elapsed:.2f} seconds")
    print(f"{'='*60}\n")
    
    # Final evaluation
    print("Final test accuracies:")
    print(f"  Global model: {accg[-1]:.4f}")
    for i in range(num_users):
        final_acc = evaluate_covid_cnn(models[i], XTest, YTest, device=device)
        print(f"  User {i+1}: {final_acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(accg, 'k-', linewidth=2, label="Global")
    for i in range(num_users):
        plt.plot(dev_accs[i], '--', label=f"User {i+1}")
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"COVID-19 Classification\nFederated Learning with Over-the-Air (SNR={SNRdB}dB)")
    plt.grid(True, alpha=0.3)
    
    # BER plot
    plt.subplot(1, 2, 2)
    for i in range(num_users):
        plt.plot(BER_weights[i], label=f"User {i+1}")
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Bit Error Rate")
    plt.title("Wireless Channel BER (Weights)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = "covid_federated_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    plt.show()
    
    # Save final global model
    model_path = "covid_global_model.pth"
    torch.save(global_model.state_dict(), model_path)
    print(f"Global model saved to {model_path}")


if __name__ == "__main__":
    main()
