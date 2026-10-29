"""audit_features.py -- per-feature audit of constant features. Read-only."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(
    r"C:\Users\Hamsini\Downloads\OmniAnomaly-master\OmniAnomaly-master\ServerMachineDataset"
)
OUT_DIR = Path(__file__).parent / "audit_results"
OUT_DIR.mkdir(exist_ok=True)

records = []
for train_path in sorted((DATA_DIR / "train").glob("machine-*.txt")):
    name = train_path.stem
    tr = np.loadtxt(train_path, delimiter=",", ndmin=2)
    te = np.loadtxt(DATA_DIR / "test" / train_path.name, delimiter=",", ndmin=2)
    assert tr.shape[1] == te.shape[1] == 38, f"unexpected width in {name}"

    for j in range(tr.shape[1]):
        # Exact constant test: max == min (more reliable than std == 0)
        tr_const = bool(tr[:, j].max() == tr[:, j].min())
        te_const = bool(te[:, j].max() == te[:, j].min())
        if tr_const and te_const:
            kind = "const_both"
        elif tr_const:
            kind = "const_train_only"
        elif te_const:
            kind = "const_test_only"
        else:
            kind = "varies_both"

        records.append({
            "machine": name,
            "feature": j,
            "kind": kind,
            "tr_const": tr_const,
            "te_const": te_const,
            "n_unique_train": int(np.unique(tr[:, j]).size),
            "train_value": float(tr[0, j]) if tr_const else np.nan,
            # for train-constant features: share of test rows that differ from that value
            "frac_test_rows_differing": (
                float((te[:, j] != tr[0, j]).mean()) if tr_const else np.nan
            ),
        })

df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "feature_audit.csv", index=False)
pd.set_option("display.width", 200)

print("=== (machine, feature) pairs by kind ===")
print(df["kind"].value_counts().to_string())

print("\n=== constant-feature counts per machine (exact max==min test) ===")
print(df.groupby("machine")[["tr_const", "te_const"]].sum().to_string())

print("\n=== constant in TRAIN but varying in TEST (dangerous for standardization) ===")
risky = df[df.kind == "const_train_only"]
if risky.empty:
    print("none")
else:
    print(risky[["machine", "feature", "train_value",
                 "frac_test_rows_differing"]].round(4).to_string(index=False))

print("\n=== per feature index: number of machines where it is constant in train ===")
counts = df[df.tr_const].groupby("feature").size()
print(counts.to_string())
print("constant in train on ALL 28 machines:", counts[counts == 28].index.tolist())
print("saved:", OUT_DIR / "feature_audit.csv")