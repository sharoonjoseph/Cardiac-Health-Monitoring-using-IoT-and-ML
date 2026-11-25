import json
import time
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(X):
    """
    Extract statistical and frequency-domain features
    from all 12 ECG leads.
    """

    feature_list = []

    for signal in X:
        features = []

        for lead in range(signal.shape[1]):
            x = signal[:, lead]

            features.extend([
                np.mean(x),
                np.std(x),
                np.min(x),
                np.max(x),
                np.median(x),
                np.ptp(x),
                skew(x),
                kurtosis(x)
            ])

            fft_values = np.abs(np.fft.rfft(x))
            features.extend([
                np.mean(fft_values),
                np.std(fft_values),
                np.max(fft_values),
                np.sum(fft_values ** 2)
            ])

        feature_list.append(features)

    return np.asarray(feature_list, dtype=np.float32)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading processed ECG datasets...")

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_validation.npy")
y_val = np.load(DATA_DIR / "y_validation.npy")

print("Training records:", len(X_train))
print("Validation records:", len(X_val))

# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

print("\nExtracting training features...")
start = time.time()

F_train = extract_features(X_train)

print("Training feature shape:", F_train.shape)

print("\nExtracting validation features...")
F_val = extract_features(X_val)

print("Validation feature shape:", F_val.shape)

print("Feature extraction time:", round(time.time() - start, 2), "seconds")

# Release large waveform arrays
del X_train, X_val

# --------------------------------------------------
# ENCODE LABELS
# --------------------------------------------------

encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_val_encoded = encoder.transform(y_val)

# --------------------------------------------------
# FEATURE SCALING
# --------------------------------------------------

print("\nScaling features...")

scaler = StandardScaler()

F_train = scaler.fit_transform(F_train)
F_val = scaler.transform(F_val)

# --------------------------------------------------
# CLASS WEIGHTS
# --------------------------------------------------

classes = np.unique(y_train_encoded)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_encoded
)

class_weights = dict(zip(classes, weights))

print("\nClass weights:")
for label, weight in zip(encoder.classes_, weights):
    print(label, ":", round(float(weight), 3))

# --------------------------------------------------
# TRAIN SVM
# --------------------------------------------------

print("\nTraining SVM model...")

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight=class_weights,
    cache_size=2000
)

start = time.time()

model.fit(F_train, y_train_encoded)

training_time = time.time() - start

print("Training time:", round(training_time, 2), "seconds")

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

print("\nEvaluating validation dataset...")

predictions = model.predict(F_val)

accuracy = accuracy_score(y_val_encoded, predictions)

macro_f1 = f1_score(
    y_val_encoded,
    predictions,
    average="macro"
)

weighted_f1 = f1_score(
    y_val_encoded,
    predictions,
    average="weighted"
)

print("\nVALIDATION RESULTS")
print("------------------------------")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("Weighted F1:", round(weighted_f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_val_encoded,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_val_encoded,
        predictions
    )
)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(model, MODEL_DIR / "svm_model.pkl")
joblib.dump(scaler, MODEL_DIR / "svm_scaler.pkl")
joblib.dump(encoder, MODEL_DIR / "svm_label_encoder.pkl")

results = {
    "model": "SVM",
    "kernel": "RBF",
    "training_records": int(len(y_train_encoded)),
    "validation_records": int(len(y_val_encoded)),
    "feature_count": int(F_train.shape[1]),
    "training_time_seconds": round(training_time, 2),
    "validation_accuracy": float(accuracy),
    "validation_macro_f1": float(macro_f1),
    "validation_weighted_f1": float(weighted_f1),
    "classes": encoder.classes_.tolist()
}

with open(MODEL_DIR / "svm_validation_results.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nSVM training completed.")
print("Saved model files in:", MODEL_DIR)