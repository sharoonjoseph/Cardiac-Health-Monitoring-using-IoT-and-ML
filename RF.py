import json
import time
import numpy as np
import joblib

from pathlib import Path
from scipy.stats import skew, kurtosis

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# ECG FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(X):
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

print("Loading ECG datasets...")

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

del X_train, X_val


# --------------------------------------------------
# LABEL ENCODING
# --------------------------------------------------

encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_val_encoded = encoder.transform(y_val)


# --------------------------------------------------
# RANDOM FOREST TRAINING
# --------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
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

print("\nRANDOM FOREST VALIDATION RESULTS")
print("--------------------------------")

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

joblib.dump(
    model,
    MODEL_DIR / "random_forest_model.pkl"
)

joblib.dump(
    encoder,
    MODEL_DIR / "random_forest_label_encoder.pkl"
)

results = {
    "model": "Random Forest",
    "training_records": int(len(y_train_encoded)),
    "validation_records": int(len(y_val_encoded)),
    "feature_count": int(F_train.shape[1]),
    "number_of_trees": 300,
    "training_time_seconds": round(training_time, 2),
    "validation_accuracy": float(accuracy),
    "validation_macro_f1": float(macro_f1),
    "validation_weighted_f1": float(weighted_f1),
    "classes": encoder.classes_.tolist()
}

with open(
    MODEL_DIR / "random_forest_validation_results.json",
    "w"
) as file:
    json.dump(results, file, indent=4)

print("\nRandom Forest training completed.")
print("Saved model files in:", MODEL_DIR)