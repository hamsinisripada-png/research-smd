"""peek_interpretation.py -- compare test_label segments with interpretation_label. Read-only."""
from pathlib import Path

import numpy as np

DATA_DIR = Path(
    r"C:\Users\Hamsini\Downloads\OmniAnomaly-master\OmniAnomaly-master\ServerMachineDataset"
)
MACHINE = "machine-1-1.txt"

labels = np.loadtxt(DATA_DIR / "test_label" / MACHINE, delimiter=",", ndmin=1)
a = (labels == 1).astype(int)
change = np.diff(np.concatenate([[0], a, [0]]))
starts = np.where(change == 1)[0]
ends = np.where(change == -1)[0]  # exclusive end

print("test_label segments:", len(starts))
for s, e in zip(starts, ends):
    print(f"  0-based {s}..{e} (end exclusive), length {e - s}; as 1-based inclusive: {s + 1}-{e}")

with open(DATA_DIR / "interpretation_label" / MACHINE, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()
print("\ninterpretation_label lines:", len(lines))
for ln in lines:
    print("  ", ln)