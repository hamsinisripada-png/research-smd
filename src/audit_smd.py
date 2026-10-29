"""audit_smd.py -- read-only data-quality audit of the Server Machine Dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(
    r"C:\Users\Hamsini\Downloads\OmniAnomaly-master\OmniAnomaly-master\ServerMachineDataset"
)
OUT_DIR = Path(__file__).parent / "audit_results"
OUT_DIR.mkdir(exist_ok=True)


def count_raw_lines(path):
    """Return (total lines, blank lines) in a text file, before any parsing."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    blank = sum(1 for line in lines if line.strip() == "")
    return len(lines), blank


def label_segments(labels):
    """Find runs of consecutive 1s. Returns (starts, ends); ends are exclusive."""
    a = (labels == 1).astype(int)
    padded = np.concatenate([[0], a, [0]])
    change = np.diff(padded)
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]
    return starts, ends


rows = []
for train_path in sorted((DATA_DIR / "train").glob("machine-*.txt")):
    test_path = DATA_DIR / "test" / train_path.name
    label_path = DATA_DIR / "test_label" / train_path.name
    assert test_path.exists(), f"missing test file for {train_path.name}"
    assert label_path.exists(), f"missing label file for {train_path.name}"

    tr_lines, tr_blank = count_raw_lines(train_path)
    te_lines, te_blank = count_raw_lines(test_path)
    tr = np.loadtxt(train_path, delimiter=",", ndmin=2)
    te = np.loadtxt(test_path, delimiter=",", ndmin=2)
    lb = np.loadtxt(label_path, delimiter=",", ndmin=1)

    starts, ends = label_segments(lb)
    seg_lengths = ends - starts

    # Shift of the test mean from the train mean, in units of train std.
    # Features with zero train std are skipped (division by zero).
    mu_tr, sd_tr = np.nanmean(tr, axis=0), np.nanstd(tr, axis=0)
    ok = sd_tr > 0
    shift = np.abs(np.nanmean(te, axis=0)[ok] - mu_tr[ok]) / sd_tr[ok]

    rows.append({
        "machine": train_path.stem,
        "train_rows": tr.shape[0],
        "test_rows": te.shape[0],
        "label_rows": lb.shape[0],
        "n_features": tr.shape[1],
        "train_raw_lines": tr_lines,
        "test_raw_lines": te_lines,
        "train_blank": tr_blank,
        "test_blank": te_blank,
        "nonfinite_train": int((~np.isfinite(tr)).sum()),
        "nonfinite_test": int((~np.isfinite(te)).sum()),
        "const_feats_train": int((np.nanstd(tr, axis=0) == 0).sum()),
        "const_feats_test": int((np.nanstd(te, axis=0) == 0).sum()),
        "label_values": str(sorted(np.unique(lb).tolist())),
        "anomaly_points": int((lb == 1).sum()),
        "anomaly_rate": round(float((lb == 1).mean()), 4),
        "n_segments": len(starts),
        "longest_segment": int(seg_lengths.max()) if len(seg_lengths) else 0,
        "max_shift_std": round(float(shift.max()), 2),
        "feats_shift_gt_3": int((shift > 3).sum()),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "data_audit.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
print(df.to_string(index=False))

print("\n=== SUMMARY CHECKS ===")
print("machines:", len(df))
print("feature counts:", sorted(df["n_features"].unique().tolist()))
print("train != test rows:", df.loc[df.train_rows != df.test_rows, "machine"].tolist())
print("test_rows != label_rows:", df.loc[df.test_rows != df.label_rows, "machine"].tolist())
print("parsed rows != raw non-blank lines (train):",
      df.loc[df.train_rows != df.train_raw_lines - df.train_blank, "machine"].tolist())
print("parsed rows != raw non-blank lines (test):",
      df.loc[df.test_rows != df.test_raw_lines - df.test_blank, "machine"].tolist())
print("machines with any non-finite values:",
      df.loc[(df.nonfinite_train + df.nonfinite_test) > 0, "machine"].tolist())
print("label value sets seen:", sorted(df["label_values"].unique().tolist()))
print("pooled anomaly rate:", round(df.anomaly_points.sum() / df.test_rows.sum(), 4))
print("saved:", OUT_DIR / "data_audit.csv")