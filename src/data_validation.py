"""data_validation.py -- steps 1-2: load a machine, check it, build one audit row.
Read-only: nothing here writes into the dataset folder."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(
    r"C:\Users\Hamsini\Downloads\OmniAnomaly-master\OmniAnomaly-master\ServerMachineDataset"
)
EXPECTED_FEATURES = 38
NEAR_CONST_MODE_FRACTION = 0.999
EXPECTED_MACHINES = 28  # train-only diagnostic threshold, not a decision rule


def sha256_of_file(path):
    """Fingerprint a file (reads it in 1 MB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_machine(machine, data_dir=DATA_DIR):
    """Load train, test and labels for one machine, e.g. 'machine-1-1'."""
    paths = {
        "train": data_dir / "train" / f"{machine}.txt",
        "test": data_dir / "test" / f"{machine}.txt",
        "label": data_dir / "test_label" / f"{machine}.txt",
    }
    for name, p in paths.items():
        assert p.exists(), f"{machine}: missing {name} file: {p}"
    train = np.loadtxt(paths["train"], delimiter=",", ndmin=2)
    test = np.loadtxt(paths["test"], delimiter=",", ndmin=2)
    labels = np.loadtxt(paths["label"], delimiter=",", ndmin=1)
    return {"paths": paths, "train": train, "test": test, "labels": labels}


def check_basic(data):
    """Return a list of failed checks. An empty list means every check passed."""
    train, test, labels = data["train"], data["test"], data["labels"]
    failures = []
    if train.shape[1] != EXPECTED_FEATURES or test.shape[1] != EXPECTED_FEATURES:
        failures.append(
            f"feature count {train.shape[1]}/{test.shape[1]} != {EXPECTED_FEATURES}"
        )
    if not (np.isfinite(train).all() and np.isfinite(test).all()):
        failures.append("non-finite values in train or test")
    if not set(np.unique(labels).tolist()) <= {0.0, 1.0}:
        failures.append("labels contain values other than 0 and 1")
    if test.shape[0] != labels.shape[0]:
        failures.append(f"test rows {test.shape[0]} != label rows {labels.shape[0]}")
    if abs(test.shape[0] - train.shape[0]) > 1:
        failures.append(
            f"train/test row difference {test.shape[0] - train.shape[0]} exceeds 1"
        )
    return failures


def label_events(labels):
    """Maximal runs of 1s as (start, end) tuples; 0-based, end exclusive."""
    a = (labels == 1).astype(int)
    change = np.diff(np.concatenate([[0], a, [0]]))
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def mode_fraction(column):
    """Share of rows taken by the most frequent value in one column."""
    _, counts = np.unique(column, return_counts=True)
    return counts.max() / column.size


def join_idx(boolean_array):
    """Turn a True/False array into '3;7;26' (0-based feature indices)."""
    return ";".join(str(i) for i in np.where(boolean_array)[0])


def audit_machine(machine, data):
    """Build one audit row (a dict) for one machine."""
    train, test, labels = data["train"], data["test"], data["labels"]
    _, group, index = machine.split("-")

    const_train = train.max(axis=0) == train.min(axis=0)  # exact constant
    const_test = test.max(axis=0) == test.min(axis=0)
    mode_frac = np.array([mode_fraction(train[:, j]) for j in range(train.shape[1])])
    near_const = (~const_train) & (mode_frac >= NEAR_CONST_MODE_FRACTION)
    stds = train.std(axis=0)
    nonconst_stds = stds[~const_train]

    events = label_events(labels)
    lengths = [e - s for s, e in events]
    failures = check_basic(data)

    return {
        "machine": machine,
        "group": int(group),
        "index": int(index),
        "train_rows": train.shape[0],
        "test_rows": test.shape[0],
        "label_rows": labels.shape[0],
        "test_minus_train": test.shape[0] - train.shape[0],
        "n_features": train.shape[1],
        "train_sha256": sha256_of_file(data["paths"]["train"]),
        "test_sha256": sha256_of_file(data["paths"]["test"]),
        "label_sha256": sha256_of_file(data["paths"]["label"]),
        "nonfinite_train": int((~np.isfinite(train)).sum()),
        "nonfinite_test": int((~np.isfinite(test)).sum()),
        "label_values_ok": bool(set(np.unique(labels).tolist()) <= {0.0, 1.0}),
        # --- train-only diagnostics (may inform decisions) ---
        "const_train_count": int(const_train.sum()),
        "const_train_idx": join_idx(const_train),
        "near_const_train_count": int(near_const.sum()),
        "near_const_train_idx": join_idx(near_const),
        "min_nonzero_train_std": float(nonconst_stds.min()) if nonconst_stds.size else float("nan"),
        # --- test-input diagnostics (descriptive only, never used for decisions) ---
        "const_test_count": int(const_test.sum()),
        "train_const_test_varying_idx": join_idx(const_train & ~const_test),
        # --- labels ---
        "anomaly_points": int((labels == 1).sum()),
        "anomaly_rate_test": float((labels == 1).mean()),
        "n_events": len(events),
        "min_event_len": min(lengths) if lengths else 0,
        "max_event_len": max(lengths) if lengths else 0,
        # --- status ---
        "checks_passed": len(failures) == 0,
        "failed_checks": "; ".join(failures),
    }


def machine_names(data_dir=DATA_DIR):
    """All machine names found in the train folder, sorted numerically."""
    names = [p.stem for p in (data_dir / "train").glob("machine-*.txt")]
    return sorted(names, key=lambda m: tuple(int(x) for x in m.split("-")[1:]))


def audit_all(data_dir=DATA_DIR):
    """Audit every machine. A failure is recorded in the row, never skipped."""
    rows = []
    for machine in machine_names(data_dir):
        try:
            data = load_machine(machine, data_dir)
            rows.append(audit_machine(machine, data))
        except AssertionError as err:
            rows.append({
                "machine": machine,
                "checks_passed": False,
                "failed_checks": f"load error: {err}",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = audit_all()
    assert len(df) == EXPECTED_MACHINES, (
        f"found {len(df)} machines, expected {EXPECTED_MACHINES}"
    )

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "data_audit_v1.csv"
    df.to_csv(out_path, index=False)

    failed = df[~df["checks_passed"]]
    print("machines audited:", len(df))
    print("machines failing checks:", len(failed))
    for _, r in failed.iterrows():
        print("  FAIL", r["machine"], "->", r["failed_checks"])

    # Re-checks against numbers from earlier audits
    print("total events:", int(df["n_events"].sum()))
    print("pooled anomaly rate:",
          round(df["anomaly_points"].sum() / df["test_rows"].sum(), 4))
    print("machines with test_minus_train == 1:",
          int((df["test_minus_train"] == 1).sum()))
    pairs = sum(len(s.split(";")) for s in df["train_const_test_varying_idx"]
                if isinstance(s, str) and s)
    print("train-constant / test-varying (machine, feature) pairs:", pairs)
    print("saved:", out_path)

    if len(failed) > 0:
        sys.exit(1)