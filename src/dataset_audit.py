
import os
import json
import time
import platform
import numpy as np
import pandas as pd

DATA_PATH = "data/raw/LSWMD.pkl"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting dataset audit...")
print("Loading the 2.1 GB dataset. Please wait.")

start = time.perf_counter()
df = pd.read_pickle(DATA_PATH)

print("\nDATASET LOADED SUCCESSFULLY")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

print("\nFirst five rows:")
print(df.drop(columns=["waferMap"], errors="ignore").head())

print("\nMissing values:")
print(df.drop(columns=["waferMap"], errors="ignore").isna().sum())

if "failureType" not in df.columns:
    raise ValueError("failureType column not found")

def decode_label(value):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size == 1:
            return decode_label(value.item())
        return str(value.tolist())

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) == 1:
            return decode_label(value[0])
        return str(value)

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    value = str(value).strip()
    return value if value else None

labels = df["failureType"].apply(decode_label)
labeled = labels.notna()

print("\nTOTAL RECORDS:", len(df))
print("LABELED RECORDS:", int(labeled.sum()))
print("UNLABELED RECORDS:", int((~labeled).sum()))

class_counts = (
    labels[labeled]
    .value_counts()
    .rename_axis("class")
    .reset_index(name="count")
)

print("\nCLASS DISTRIBUTION:")
print(class_counts.to_string(index=False))

class_counts.to_csv(
    f"{OUTPUT_DIR}/class_counts.csv",
    index=False
)

if "trianTestLabel" in df.columns:
    print("\nORIGINAL SPLIT DISTRIBUTION:")
    print(df["trianTestLabel"].value_counts(dropna=False))

    df["trianTestLabel"].value_counts(
        dropna=False
    ).to_csv(
        f"{OUTPUT_DIR}/original_split_counts.csv",
        header=["count"]
    )

if "waferMap" in df.columns:
    print("\nWAFER MAP COLUMN: AVAILABLE")

    shapes = df["waferMap"].apply(
        lambda x: tuple(np.asarray(x).shape)
    )

    shape_counts = (
        shapes.value_counts()
        .rename_axis("shape")
        .reset_index(name="count")
    )

    shape_counts["shape"] = shape_counts["shape"].astype(str)
    shape_counts.to_csv(
        f"{OUTPUT_DIR}/map_shape_counts.csv",
        index=False
    )

    print("Distinct map shapes:", shapes.nunique())
    print(shape_counts.head(20).to_string(index=False))
else:
    print("\nWARNING: waferMap column is missing")

if "lotName" in df.columns:
    print("\nUnique lots:", df["lotName"].nunique())

report = {
    "dataset_path": DATA_PATH,
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "column_names": df.columns.tolist(),
    "labeled_records": int(labeled.sum()),
    "unlabeled_records": int((~labeled).sum()),
    "unique_classes": int(labels[labeled].nunique()),
    "has_waferMap": "waferMap" in df.columns,
    "unique_lots": int(df["lotName"].nunique())
    if "lotName" in df.columns else None,
    "python_version": platform.python_version(),
    "audit_seconds": round(time.perf_counter() - start, 2)
}

with open(
    f"{OUTPUT_DIR}/dataset_audit.json", "w"
) as file:
    json.dump(report, file, indent=2)

print("\nAUDIT SUMMARY:")
print(json.dumps(report, indent=2))
print("\nAudit files saved in the results folder.")
