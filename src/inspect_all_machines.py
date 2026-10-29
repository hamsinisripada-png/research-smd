
from pathlib import Path

import numpy as np
import pandas as pd

data_dir = Path(__file__).resolve().parent

train_dir = data_dir / "train"
test_dir = data_dir / "test"
label_dir = data_dir / "test_label"

train_files = sorted(train_dir.glob("*.txt"))

print("Number of training machine files:", len(train_files))
print()

results = []

for train_path in train_files:
    machine = train_path.name

    test_path = test_dir / machine
    label_path = label_dir / machine

    if not test_path.exists() or not label_path.exists():
        print("Missing matching file:", machine)
        continue

    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)
    labels = np.loadtxt(label_path, delimiter=",")

    if len(test) != len(labels):
        print("LABEL LENGTH MISMATCH:", machine)

    results.append({
        "machine": machine,
        "train_rows": len(train),
        "test_rows": len(test),
        "features": train.shape[1],
        "anomaly_points": int(np.sum(labels == 1)),
        "anomaly_rate_pct": round(np.mean(labels == 1) * 100, 2),
        "label_count_matches": len(test) == len(labels),
    })

summary = pd.DataFrame(results)

print(summary.to_string(index=False))

print("\nAll test rows match label counts:",
      summary["label_count_matches"].all())

print("\nDistinct feature counts:",
      sorted(summary["features"].unique()))

print("\n--- INVESTIGATING LENGTH DIFFERENCES ---")

for train_path in sorted(train_dir.glob("*.txt")):
    machine = train_path.name

    test_path = test_dir / machine
    label_path = label_dir / machine

    if not test_path.exists() or not label_path.exists():
        continue

    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)
    labels = np.loadtxt(label_path, delimiter=",")

    difference = len(test) - len(train)

    if difference != 0:
        print(
            f"{machine}: "
            f"train={len(train)}, "
            f"test={len(test)}, "
            f"labels={len(labels)}, "
            f"test-train={difference}"
        )

print("\nCheck complete.")
