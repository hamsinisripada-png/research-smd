"""compare_interpretation_v2.py -- order-independent comparison of test_label runs
and interpretation_label ranges. Read-only."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(
    r"C:\Users\Hamsini\Downloads\OmniAnomaly-master\OmniAnomaly-master\ServerMachineDataset"
)
OUT_DIR = Path(__file__).parent / "audit_results"
OUT_DIR.mkdir(exist_ok=True)


def label_intervals(labels):
    """Runs of consecutive 1s as (start, end) tuples, 0-based, end exclusive."""
    a = (labels == 1).astype(int)
    change = np.diff(np.concatenate([[0], a, [0]]))
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def parse_interpretation(path):
    """Lines look like 'start-end:dim,dim,...'. Returns [(start, end), ...]."""
    ranges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            if line.strip() == "":
                continue
            start, end = line.split(":")[0].split("-")
            ranges.append((int(start), int(end)))
    return ranges


rows, detail = [], []
for label_path in sorted((DATA_DIR / "test_label").glob("machine-*.txt")):
    name = label_path.stem
    interp_path = DATA_DIR / "interpretation_label" / label_path.name
    assert interp_path.exists(), f"missing interpretation file for {name}"

    labels = np.loadtxt(label_path, delimiter=",", ndmin=1)
    label_iv = label_intervals(labels)
    interp_iv = parse_interpretation(interp_path)

    # Fail loudly if a range does not fit inside the label sequence
    for s, e in interp_iv:
        assert 0 <= s < e <= len(labels), f"{name}: bad range {s}-{e}"

    # 1. exact interval comparison, ignoring order
    only_label = sorted(set(label_iv) - set(interp_iv))
    only_interp = sorted(set(interp_iv) - set(label_iv))
    for s, e in only_label:
        detail.append({"machine": name, "where": "only_in_test_label", "start": s, "end": e})
    for s, e in only_interp:
        detail.append({"machine": name, "where": "only_in_interpretation", "start": s, "end": e})

    # 2. point-level comparison via a 0/1 mask of the interpretation ranges
    mask = np.zeros(len(labels), dtype=int)
    for s, e in interp_iv:
        mask[s:e] = 1

    # 3. do the interpretation ranges touch or overlap each other?
    srt = sorted(interp_iv)
    abutting = sum(1 for a, b in zip(srt, srt[1:]) if a[1] == b[0])
    overlapping = sum(1 for a, b in zip(srt, srt[1:]) if a[1] > b[0])

    rows.append({
        "machine": name,
        "label_runs": len(label_iv),
        "interp_ranges": len(interp_iv),
        "interp_in_order": interp_iv == srt,
        "exact_matches": len(set(label_iv) & set(interp_iv)),
        "only_in_label": len(only_label),
        "only_in_interp": len(only_interp),
        "pts_both": int(((labels == 1) & (mask == 1)).sum()),
        "pts_only_label": int(((labels == 1) & (mask == 0)).sum()),
        "pts_only_interp": int(((labels == 0) & (mask == 1)).sum()),
        "interp_abutting": abutting,
        "interp_overlapping": overlapping,
    })

sdf = pd.DataFrame(rows)
ddf = pd.DataFrame(detail)
sdf.to_csv(OUT_DIR / "interp_v2_summary.csv", index=False)
ddf.to_csv(OUT_DIR / "interp_v2_detail.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
print(sdf.to_string(index=False))
print("\n=== intervals without an exact counterpart ===")
print(ddf.to_string(index=False) if len(ddf) else "none")
print("\nmachines with interpretation lines out of order:",
      sdf.loc[~sdf.interp_in_order, "machine"].tolist())
print("machines with any unmatched interval:",
      sdf.loc[(sdf.only_in_label + sdf.only_in_interp) > 0, "machine"].tolist())
print("total points only in test_label:", int(sdf.pts_only_label.sum()))
print("total points only in interpretation ranges:", int(sdf.pts_only_interp.sum()))
print("saved to:", OUT_DIR)