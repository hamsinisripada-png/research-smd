"""compare_interpretation.py -- compare test_label runs with interpretation_label
ranges for all machines. Read-only."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(
    r"C:\Users\Hamsini\Downloads\OmniAnomaly-master\OmniAnomaly-master\ServerMachineDataset"
)
OUT_DIR = Path(__file__).parent / "audit_results"
OUT_DIR.mkdir(exist_ok=True)


def label_segments(labels):
    """Runs of consecutive 1s: (starts, ends), 0-based, ends exclusive."""
    a = (labels == 1).astype(int)
    change = np.diff(np.concatenate([[0], a, [0]]))
    return np.where(change == 1)[0], np.where(change == -1)[0]


def parse_interpretation(path):
    """Lines look like 'start-end:dim,dim,...'. Returns [(start, end), ...]."""
    ranges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            if line.strip() == "":
                continue
            span = line.split(":")[0]
            start, end = span.split("-")
            ranges.append((int(start), int(end)))
    return ranges


summary, mismatches = [], []
for label_path in sorted((DATA_DIR / "test_label").glob("machine-*.txt")):
    name = label_path.stem
    interp_path = DATA_DIR / "interpretation_label" / label_path.name
    assert interp_path.exists(), f"missing interpretation file for {name}"

    labels = np.loadtxt(label_path, delimiter=",", ndmin=1)
    starts, ends = label_segments(labels)
    ranges = parse_interpretation(interp_path)

    same_count = len(starts) == len(ranges)
    n_bad = None
    if same_count:  # pair segments by order only when counts agree
        n_bad = 0
        for i in range(len(starts)):
            ls, le = int(starts[i]), int(ends[i])
            is_, ie = ranges[i]
            if ls != is_ or le != ie:
                n_bad += 1
                mismatches.append({
                    "machine": name, "segment": i,
                    "label_start": ls, "label_end": le,
                    "interp_start": is_, "interp_end": ie,
                    "start_diff": ls - is_, "end_diff": le - ie,
                })
    summary.append({
        "machine": name,
        "label_segments": len(starts),
        "interp_ranges": len(ranges),
        "same_count": same_count,
        "mismatched_segments": n_bad,
    })

sdf = pd.DataFrame(summary)
mdf = pd.DataFrame(mismatches)
sdf.to_csv(OUT_DIR / "interp_summary.csv", index=False)
mdf.to_csv(OUT_DIR / "interp_mismatches.csv", index=False)

pd.set_option("display.width", 200)
print(sdf.to_string(index=False))
print("\n=== segment-level mismatches ===")
print(mdf.to_string(index=False) if len(mdf) else "none")
print("\nmachines with different segment counts:",
      sdf.loc[~sdf.same_count, "machine"].tolist())
print("total mismatched segments:", len(mdf))
print("saved to:", OUT_DIR)
print("total label segments:", sdf["label_segments"].sum())