
import argparse
import json
import os
import numpy as np
import pandas as pd

def get_label(value):
    if isinstance(value, str):
        return value
    try:
        for item in np.asarray(value, dtype=object).ravel():
            if isinstance(item, str):
                return item
    except Exception:
        pass
    return None

parser = argparse.ArgumentParser()
parser.add_argument("--pkl", default="data/raw/LSWMD.pkl")
parser.add_argument("--split-json", default="docs/TrackE_split_lot_lists.json")
parser.add_argument("--out-dir", default="Split_IDs")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

print("Loading dataset. Please wait...", flush=True)
df = pd.read_pickle(args.pkl)

split_col = "trainTestLabel" if "trainTestLabel" in df.columns else "trianTestLabel"
df["label"] = df["failureType"].map(get_label)
df["split"] = df[split_col].map(get_label)
df["sample_id"] = np.arange(len(df))

with open(args.split_json, encoding="utf-8") as file:
    data = json.load(file)

lots = data.get("E1E3_val_lots")
if lots is None:
    raise ValueError("Could not find E1E3_val_lots in the JSON. Check its structure.")

validation_lots = set(map(str, lots))
labelled = df[df["label"].notna()].copy()

train = labelled[labelled["split"].astype(str).str.casefold().eq("training")]
test = labelled[labelled["split"].astype(str).str.casefold().eq("test")]

missing = validation_lots - set(train["lotName"].astype(str))
if missing:
    raise ValueError(
        f"{len(missing)} validation lots were not found in training data. "
        "Check the correct lot-list JSON before proceeding."
    )

val_mask = train["lotName"].astype(str).isin(validation_lots)
fit = train[~val_mask].copy()
val = train[val_mask].copy()

assert set(fit["sample_id"]).isdisjoint(set(val["sample_id"]))
assert set(fit["sample_id"]).isdisjoint(set(test["sample_id"]))
assert set(val["sample_id"]).isdisjoint(set(test["sample_id"]))

columns = ["sample_id", "lotName", "waferIndex", "label", "split"]
for filename, subset in [
    ("train_ids.csv", fit),
    ("val_ids.csv", val),
    ("test_ids.csv", test),
]:
    subset[columns].to_csv(
        os.path.join(args.out_dir, filename), index=False
    )
    print(f"Saved {filename}: {len(subset)} samples", flush=True)

manifest = {
    "validation_rule": "Instructor-issued E1E3_val_lots list",
    "fit_count": len(fit),
    "validation_count": len(val),
    "test_count": len(test),
    "fit_validation_lot_overlap": len(
        set(fit["lotName"].astype(str)) & set(val["lotName"].astype(str))
    ),
}

with open(os.path.join(args.out_dir, "split_manifest.json"), "w", encoding="utf-8") as file:
    json.dump(manifest, file, indent=2)

print(json.dumps(manifest, indent=2))
print("Split file generation completed.", flush=True)
