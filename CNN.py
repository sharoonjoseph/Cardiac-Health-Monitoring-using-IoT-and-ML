import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 6

print("Using device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class ECGDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(
            X.transpose(0, 2, 1),
            dtype=torch.float32
        )

        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# --------------------------------------------------
# 1D CNN ARCHITECTURE
# --------------------------------------------------

class ECGCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading ECG datasets...")

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_validation.npy")
y_val = np.load(DATA_DIR / "y_validation.npy")

encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_val_encoded = encoder.transform(y_val)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Classes:", encoder.classes_)

# --------------------------------------------------
# TRAINING-ONLY NORMALIZATION
# --------------------------------------------------

print("\nCalculating training normalization parameters...")

# One mean and standard deviation per lead,
# calculated exclusively from training data.
lead_mean = X_train.mean(axis=(0, 1), dtype=np.float64)
lead_std = X_train.std(axis=(0, 1), dtype=np.float64)

lead_std = np.where(lead_std < 1e-8, 1.0, lead_std)

X_train = (
    (X_train - lead_mean.reshape(1, 1, 12))
    / lead_std.reshape(1, 1, 12)
).astype(np.float32)

X_val = (
    (X_val - lead_mean.reshape(1, 1, 12))
    / lead_std.reshape(1, 1, 12)
).astype(np.float32)

# --------------------------------------------------
# DATALOADERS
# --------------------------------------------------

train_dataset = ECGDataset(X_train, y_train_encoded)
val_dataset = ECGDataset(X_val, y_val_encoded)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

# --------------------------------------------------
# MODEL AND CLASS WEIGHTS
# --------------------------------------------------

num_classes = len(encoder.classes_)

model = ECGCNN(num_classes).to(DEVICE)

classes = np.arange(num_classes)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_encoded
)

class_weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# --------------------------------------------------
# VALIDATION FUNCTION
# --------------------------------------------------

def evaluate(loader):

    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():

        for signals, labels in loader:

            signals = signals.to(DEVICE, non_blocking=True)

            outputs = model(signals)

            predictions = torch.argmax(outputs, dim=1)

            actual.extend(labels.numpy())
            predicted.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(actual, predicted)

    macro_f1 = f1_score(
        actual,
        predicted,
        average="macro"
    )

    weighted_f1 = f1_score(
        actual,
        predicted,
        average="weighted"
    )

    return accuracy, macro_f1, weighted_f1, actual, predicted


# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------

best_macro_f1 = -1
best_epoch = 0
epochs_without_improvement = 0
best_state = None

history = []

print("\nStarting 1D CNN training...")

start_time = time.time()

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for signals, labels in train_loader:

        signals = signals.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(signals)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * signals.size(0)

    train_loss = total_loss / len(train_dataset)

    accuracy, macro_f1, weighted_f1, _, _ = evaluate(val_loader)

    scheduler.step(macro_f1)

    history.append({
        "epoch": epoch + 1,
        "training_loss": train_loss,
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
        "validation_weighted_f1": weighted_f1
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro F1: {macro_f1:.4f} | "
        f"Weighted F1: {weighted_f1:.4f}"
    )

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1
        best_epoch = epoch + 1
        best_state = copy.deepcopy(model.state_dict())

        epochs_without_improvement = 0

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


training_time = time.time() - start_time

# --------------------------------------------------
# RESTORE BEST MODEL
# --------------------------------------------------

model.load_state_dict(best_state)

accuracy, macro_f1, weighted_f1, actual, predicted = evaluate(val_loader)

print("\nFINAL CNN VALIDATION RESULTS")
print("--------------------------------")
print("Best epoch:", best_epoch)
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("Weighted F1:", round(weighted_f1, 4))
print("Training time:", round(training_time, 2), "seconds")

print("\nClassification Report:")

print(
    classification_report(
        actual,
        predicted,
        labels=np.arange(num_classes),
        target_names=encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(confusion_matrix(actual, predicted))

# --------------------------------------------------
# SAVE CHECKPOINT
# --------------------------------------------------

checkpoint = {
    "model_state_dict": model.state_dict(),
    "classes": encoder.classes_.tolist(),
    "lead_mean": lead_mean.tolist(),
    "lead_std": lead_std.tolist(),
    "input_leads": 12,
    "input_samples": 1000,
    "best_epoch": best_epoch
}

torch.save(
    checkpoint,
    MODEL_DIR / "cnn_model.pth"
)

with open(MODEL_DIR / "cnn_training_history.json", "w") as file:
    json.dump(history, file, indent=4)

results = {
    "model": "1D CNN",
    "training_records": len(train_dataset),
    "validation_records": len(val_dataset),
    "epochs_completed": len(history),
    "best_epoch": best_epoch,
    "training_time_seconds": round(training_time, 2),
    "validation_accuracy": float(accuracy),
    "validation_macro_f1": float(macro_f1),
    "validation_weighted_f1": float(weighted_f1),
    "classes": encoder.classes_.tolist()
}

with open(MODEL_DIR / "cnn_validation_results.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nCNN training completed successfully.")
print("Saved checkpoint in:", MODEL_DIR)