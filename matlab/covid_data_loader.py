"""
Load COVID-19 Radiography Dataset for federated learning with over-the-air aggregation.
Partitions image data across multiple users (simulating different hospitals/sites).
"""
import numpy as np
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

DATASET_ROOT = '/home/kali/Desktop/QNN/COVID-19_Radiography_Dataset'

def load_covid_images(dataset_root=None, img_size=(224, 224), num_classes=4):
    """
    Load COVID-19 Radiography Dataset images and labels.
    
    Args:
        dataset_root: Path to COVID-19_Radiography_Dataset folder
        img_size: Target image size (height, width)
        num_classes: Number of classes to load (2 or 4)
                    2: binary (COVID vs Normal)
                    4: multiclass (COVID, Lung_Opacity, Normal, Viral Pneumonia)
    
    Returns:
        images: numpy array of shape (N, height, width, channels)
        labels: numpy array of shape (N,) with class indices
        class_names: list of class names
    """
    if dataset_root is None:
        dataset_root = DATASET_ROOT
    dataset_root = Path(dataset_root)
    
    if num_classes == 2:
        # Binary classification: COVID vs Normal
        class_folders = ['COVID', 'Normal']
    elif num_classes == 4:
        # Multiclass: all 4 categories
        class_folders = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    else:
        raise ValueError("num_classes must be 2 or 4")
    
    images_list = []
    labels_list = []
    
    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
        transforms.ToTensor(),
    ])
    
    print(f"Loading {num_classes}-class COVID-19 dataset from {dataset_root}...")
    
    for class_idx, class_name in enumerate(class_folders):
        class_path = dataset_root / class_name / 'images'
        if not class_path.exists():
            print(f"Warning: {class_path} not found, skipping...")
            continue
        
        image_files = sorted(class_path.glob('*.png'))
        print(f"  {class_name}: {len(image_files)} images")
        
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img)
                # Convert to numpy (C, H, W) -> (H, W, C)
                img_array = img_tensor.permute(1, 2, 0).numpy()
                images_list.append(img_array)
                labels_list.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    images = np.stack(images_list, axis=0).astype(np.float32)
    labels = np.array(labels_list, dtype=np.int64)
    
    print(f"Loaded {len(images)} images with shape {images.shape}")
    return images, labels, class_folders


# Function: partition_data_federated - Helper routine for partition data federated logic.
# Parameters: `images` is images input value; `labels` is labels input value; `num_users` is num users input value; `test_fraction` is test fraction input value; `seed` is seed input value.
def partition_data_federated(images, labels, num_users=4, test_fraction=0.2, seed=42):
    """
    Partition data across users for federated learning (IID partitioning).
    
    Args:
        images: numpy array (N, H, W, C)
        labels: numpy array (N,)
        num_users: number of federated users/clients
        test_fraction: fraction of data to reserve for test set
        seed: random seed for reproducibility
    
    Returns:
        XTrain: list of num_users arrays, each (samples_i, H, W, C)
        YTrain: list of num_users arrays, each (samples_i,)
        XTest: array (test_samples, H, W, C)
        YTest: array (test_samples,)
    """
    rng = np.random.default_rng(seed)
    
    # Shuffle data
    indices = np.arange(len(images))
    rng.shuffle(indices)
    images = images[indices]
    labels = labels[indices]
    
    # Split into train and test
    n_test = int(len(images) * test_fraction)
    n_train = len(images) - n_test
    
    XTest = images[:n_test]
    YTest = labels[:n_test]
    XTrain_all = images[n_test:]
    YTrain_all = labels[n_test:]
    
    # Partition training data across users (IID - round-robin)
    XTrain = [[] for _ in range(num_users)]
    YTrain = [[] for _ in range(num_users)]
    
    for i, (img, lbl) in enumerate(zip(XTrain_all, YTrain_all)):
        user_idx = i % num_users
        XTrain[user_idx].append(img)
        YTrain[user_idx].append(lbl)
    
    # Convert to numpy arrays
    XTrain = [np.stack(user_data, axis=0) for user_data in XTrain]
    YTrain = [np.array(user_labels, dtype=np.int64) for user_labels in YTrain]
    
    print(f"\nData partitioning:")
    print(f"  Test set: {len(XTest)} samples")
    for i in range(num_users):
        print(f"  User {i+1}: {len(XTrain[i])} samples")
    
    return XTrain, YTrain, XTest, YTest


def load_covid_federated(dataset_root=None, num_users=4, img_size=(224, 224), 
                         num_classes=4, test_fraction=0.2):
    """
    Convenience function to load and partition COVID-19 data in one call.
    
    Returns:
        XTrain: list of num_users image arrays
        YTrain: list of num_users label arrays
        XTest: test image array
        YTest: test label array
        class_names: list of class names
    """
    if dataset_root is None:
        # Default to workspace root
        dataset_root = Path(__file__).parent.parent / 'COVID-19_Radiography_Dataset'
    
    images, labels, class_names = load_covid_images(
        dataset_root, img_size=img_size, num_classes=num_classes
    )
    
    XTrain, YTrain, XTest, YTest = partition_data_federated(
        images, labels, num_users=num_users, test_fraction=test_fraction
    )
    
    return XTrain, YTrain, XTest, YTest, class_names


if __name__ == "__main__":
    # Test data loading
    XTrain, YTrain, XTest, YTest, class_names = load_covid_federated(
        num_users=4, num_classes=4
    )
    print(f"\nClass names: {class_names}")
    print(f"Image shape: {XTrain[0].shape}")
    print(f"Label distribution in test set:", np.bincount(YTest))
