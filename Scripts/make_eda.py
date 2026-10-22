import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PKL = r"data\raw\LSWMD.pkl"
OUT = r"EDA"
os.makedirs(OUT, exist_ok=True)

def first_string(value):
    if isinstance(value, str):
        return value
    try:
        for item in np.asarray(value, dtype=object).ravel():
            if isinstance(item, str):
                return item
    except Exception:
        pass
    return "Unknown"

print("Loading dataset; this may take several minutes...", flush=True)
df = pd.read_pickle(PKL)
labels = df["failureType"].map(first_string)
counts = labels.value_counts(dropna=False)

plt.figure(figsize=(10, 5))
counts.sort_values(ascending=False).plot(kind="bar")
plt.title("WM-811K Wafer Map Label Distribution")
plt.xlabel("Label")
plt.ylabel("Number of records")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "class_distribution.png"), dpi=180)
plt.close()
print("Saved EDA/class_distribution.png", flush=True)

wanted = ["Edge-Ring", "Edge-Loc", "Center", "Loc", "Scratch", "Random", "Donut", "Near-full"]
sample_df = df[labels.isin(wanted)].copy()
sample_df["_label"] = labels.loc[sample_df.index]
sample_df = sample_df[sample_df["waferMap"].notna()].head(12)

fig, axes = plt.subplots(3, 4, figsize=(10, 8))
for ax, (_, row) in zip(axes.ravel(), sample_df.iterrows()):
    ax.imshow(np.asarray(row["waferMap"]), interpolation="nearest")
    ax.set_title(str(row["_label"]))
    ax.axis("off")
for ax in axes.ravel()[len(sample_df):]:
    ax.axis("off")
plt.suptitle("Example Labelled Wafer Maps")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "sample_wafer_maps.png"), dpi=180)
plt.close()
print("Saved EDA/sample_wafer_maps.png", flush=True)

summary = pd.DataFrame({"label": counts.index.astype(str), "count": counts.values})
summary.to_csv(os.path.join(OUT, "class_counts.csv"), index=False)
print("Saved EDA/class_counts.csv", flush=True)
print("EDA completed.", flush=True)
