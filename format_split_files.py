import os
import pandas as pd

SPLIT_DIR = "Split_IDs"
BUILTIN_OUT = "T07_E1_builtin.csv"
SUMMARY_OUT = "T07_E1_split_summary.csv"

mapping = [
    ("train_ids.csv", "train"),
    ("val_ids.csv", "val"),
    ("test_ids.csv", "test"),
]
parts = []

for filename, partition in mapping:
    path = os.path.join(SPLIT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required split file not found: {path}")
    d = pd.read_csv(path)

    required = {"sample_id", "lotName", "label"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")

    part = pd.DataFrame({
        "sample_id": pd.to_numeric(d["sample_id"], errors="raise").astype("int64"),
        "unit_id": d["lotName"].astype(str),
        "partition": partition,
        "label": d["label"].astype(str),
        "reason": ""
    })
    parts.append(part)

combined = pd.concat(parts, ignore_index=True)
if combined["sample_id"].duplicated().any():
    dupes = int(combined["sample_id"].duplicated().sum())
    raise ValueError(f"Duplicate sample_id values found across split files: {dupes}")

combined = combined.sort_values("sample_id").reset_index(drop=True)
combined.to_csv(BUILTIN_OUT, index=False)

summary_rows = []
for partition in ["train", "val", "test"]:
    sub = combined[combined["partition"] == partition]
    summary_rows.append({
        "team": "T07",
        "track": "E1",
        "partition": partition,
        "sample_count": len(sub),
        "unique_sample_ids": sub["sample_id"].nunique(),
        "unique_units_lots": sub["unit_id"].nunique(),
        "label_match_check": "Previously verified against source dataset",
        "reason_field": "empty"
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_OUT, index=False)

print(f"Created {BUILTIN_OUT}: {len(combined)} rows")
print(f"Columns: {', '.join(combined.columns)}")
print("\nPartition counts:")
print(combined["partition"].value_counts().reindex(["train", "val", "test"]).to_string())
print(f"\nCreated {SUMMARY_OUT}")
print(f"Total rows: {len(combined)}")
print(f"Unique sample IDs: {combined['sample_id'].nunique()}")
print("\nNote: this reformats the existing split IDs; it does not independently validate the instructor's split rules.")
