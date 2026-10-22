import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report

DATA_PATH = r"data\raw\LSWMD.pkl"
SPLIT_DIR = r"Split_IDs"
OUT_DIR = r"results\baseline"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading dataset...", flush=True)
df = pd.read_pickle(DATA_PATH)

def get_label(value):
    try:
        return str(np.asarray(value, dtype=object).ravel()[0])
    except Exception:
        return "none"

df["label"] = df["failureType"].apply(get_label)

def extract_features(wafer_map):
    arr = np.asarray(wafer_map, dtype=np.uint8)
    total = arr.size
    if total == 0:
        return np.zeros(12, dtype=np.float32)

    values, counts = np.unique(arr, return_counts=True)
    proportions = np.zeros(3, dtype=np.float32)
    for value, count in zip(values, counts):
        if value < 3:
            proportions[int(value)] = count / total

    return np.array([
        arr.shape[0],
        arr.shape[1],
        proportions[0],
        proportions[1],
        proportions[2],
        np.mean(arr),
        np.std(arr),
        np.mean(arr == 0),
        np.mean(arr == 1),
        np.mean(arr == 2),
        np.mean(arr[0, :]),
        np.mean(arr[-1, :])
    ], dtype=np.float32)

print("Extracting features...", flush=True)
features = np.vstack(df["waferMap"].apply(extract_features).to_numpy())

results = {}
for split_name, filename in [
    ("train", "train_ids.csv"),
    ("validation", "val_ids.csv"),
    ("test", "test_ids.csv")
]:
    split_df = pd.read_csv(os.path.join(SPLIT_DIR, filename))
    ids = split_df["sample_id"].astype(int).to_numpy()
    results[split_name] = (ids, split_df["label"].astype(str).to_numpy())

X_train, y_train = results["train"]
X_val, y_val = results["validation"]
X_test, y_test = results["test"]

print("Preparing training data...", flush=True)
X_train_features = features[X_train]
X_val_features = features[X_val]
X_test_features = features[X_test]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=4
)

print("Training Random Forest...", flush=True)
start = time.time()
model.fit(X_train_features, y_train)
training_seconds = time.time() - start

print("Evaluating validation split...", flush=True)
val_predictions = model.predict(X_val_features)

metrics = {
    "accuracy": accuracy_score(y_val, val_predictions),
    "macro_f1": f1_score(y_val, val_predictions, average="macro", zero_division=0),
    "balanced_accuracy": balanced_accuracy_score(y_val, val_predictions),
    "training_seconds": training_seconds
}

print("\nValidation metrics:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nValidation classification report:")
print(classification_report(y_val, val_predictions, zero_division=0))

pd.DataFrame([metrics]).to_csv(
    os.path.join(OUT_DIR, "validation_metrics.csv"), index=False
)
pd.DataFrame({
    "sample_id": X_val,
    "true_label": y_val,
    "predicted_label": val_predictions
}).to_csv(os.path.join(OUT_DIR, "validation_predictions.csv"), index=False)

import joblib
joblib.dump(model, os.path.join(OUT_DIR, "random_forest_baseline.joblib"))

print("\nBaseline saved in results/baseline", flush=True)