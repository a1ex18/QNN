import keras
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import build_federated_datasets

MODEL_PATH = "federated_pipeline/global_model.h5"
BATCH_SIZE = 16

_, _, test_ds = build_federated_datasets()

# Function: evaluate_model - Helper routine for evaluate model logic.
# Parameters: `model` is model instance used for training/inference; `test_ds` is test ds input value.
def evaluate_model(model, test_ds):
    y_true = []
    y_pred = []
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x)
        y_true.extend(batch_y.numpy().astype(int))
        y_pred.extend((preds.flatten() > 0.5).astype(int))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm

# Function: plot_confusion_matrix - Helper routine for plot confusion matrix logic.
# Parameters: `cm` is cm input value; `labels` is labels input value; `out_path` is filesystem path input.
def plot_confusion_matrix(cm, labels, out_path):
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Function: main - Helper routine for main logic.
# Parameters: none.
def main():
    model = keras.models.load_model(MODEL_PATH)
    # Print weights and biases for each layer
    print("\n--- Global Model Weights and Biases ---")
    weights = model.get_weights()
    for i, w in enumerate(weights):
        print(f"Layer {i} weights/biases: shape={w.shape}, mean={np.mean(w):.4f}, std={np.std(w):.4f}")
        print(w)
        if len(w.shape) == 1:
            print(f"Layer {i} BIASES: {w}\n")
    print("--- End Weights and Biases ---\n")
    acc, prec, rec, f1, cm = evaluate_model(model, test_ds)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    plot_confusion_matrix(cm, labels=["Normal", "COVID"], out_path="federated_pipeline/confusion_matrix.png")
    with open("federated_pipeline/eval_report.txt", "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
    print("Saved confusion matrix and report to federated_pipeline/")

if __name__ == "__main__":
    main()
