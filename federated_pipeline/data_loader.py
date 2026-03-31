import os
import random
from typing import Dict, List, Tuple
import tensorflow as tf
import numpy as np


DATASET_ROOT = "/home/kali/Desktop/QNN/COVID-19_Radiography_Dataset"
CLASS_MAPPING = {"COVID": 1, "Normal": 0}
IMAGE_SIZE = (224, 224)

def list_image_paths() -> List[Tuple[str, int]]:
    pairs = []
    for class_name, label in CLASS_MAPPING.items():
        img_dir = os.path.join(DATASET_ROOT, f"{class_name}/images")
        if not os.path.isdir(img_dir):
            continue
        for f in os.listdir(img_dir):
            if f.lower().endswith('.png'):
                pairs.append((os.path.join(img_dir, f), label))
    random.shuffle(pairs)
    return pairs

# Function: train_val_test_split - Helper routine for train val test split logic.
# Parameters: `pairs` is pairs input value; `int]]` is int]] input value; `train_ratio` is train ratio input value; `val_ratio` is val ratio input value.
def train_val_test_split(pairs: List[Tuple[str,int]], train_ratio=0.7, val_ratio=0.15):
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = pairs[:n_train]
    val = pairs[n_train:n_train+n_val]
    test = pairs[n_train+n_val:]
    return train, val, test

def partition_clients(pairs: List[Tuple[str,int]], num_clients: int) -> Dict[str, List[Tuple[str,int]]]:
    # Simple even partition (IID); could introduce non-IID later
    shards = {f"client_{i+1}": [] for i in range(num_clients)}
    for idx, item in enumerate(pairs):
        shards[f"client_{idx % num_clients + 1}"].append(item)
    return shards

# Function: preprocess_image - Helper routine for preprocess image logic.
# Parameters: `path` is filesystem path input; `label` is label input value; `augment` is augment input value.
def preprocess_image(path, label, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        # tf.image.random_zoom is not available in TF 2.x, skip or implement custom if needed
    return img, tf.cast(label, tf.float32)


# Function: make_dataset - Helper routine for make dataset logic.
# Parameters: `pairs` is pairs input value; `int]]` is int]] input value; `batch_size` is batch size input value; `augment` is augment input value.
def make_dataset(pairs: List[Tuple[str,int]], batch_size=32, augment=False):
    paths = [p for p,_ in pairs]
    labels = [l for _,l in pairs]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if augment:
        ds = ds.shuffle(len(pairs))
        ds = ds.map(lambda path, label: preprocess_image(path, label, augment=True), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda path, label: preprocess_image(path, label, augment=False), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# Function: build_federated_datasets - Helper routine for build federated datasets logic.
# Parameters: `num_clients` is num clients input value; `batch_size` is batch size input value.
def build_federated_datasets(num_clients=8, batch_size=32):
    pairs = list_image_paths()
    train, val, test = train_val_test_split(pairs)
    client_partitions = partition_clients(train, num_clients)
    client_datasets = {}
    for cid, cpairs in client_partitions.items():
        client_datasets[cid] = make_dataset(cpairs, batch_size=batch_size, augment=True)
    val_ds = make_dataset(val, batch_size=batch_size, augment=False)
    test_ds = make_dataset(test, batch_size=batch_size, augment=False)
    return client_datasets, val_ds, test_ds

if __name__ == "__main__":
    cds, vds, tds = build_federated_datasets()
    print({k: v for k,v in cds.items()})
    print(vds, tds)
