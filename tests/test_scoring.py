"""test_scoring.py -- synthetic unit tests, plus training-rows-only checks on real data."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_validation import load_machine  # noqa: E402
from scoring import alarms, fit_scaler, score_rows, threshold_at  # noqa: E402

# --- synthetic: known z-scores ---
train = np.array([[0, 10], [0, 20], [2, 10], [2, 20]], dtype=float)  # mean [1,15], std [1,5]
sc = fit_scaler(train)
assert sc["idx"].tolist() == [0, 1]
assert np.allclose(sc["mean"], [1, 15]) and np.allclose(sc["std"], [1, 5])
X = np.array([[3, 40],   # z = [2, 5] -> score 5, feature 1
              [5, 15],   # z = [4, 0] -> score 4, feature 0
              [4, 30]])  # z = [3, 3] -> score 3, tie: lowest index (0) wins
scores, arg = score_rows(X, sc)
assert np.allclose(scores, [5, 4, 3]), scores
assert arg.tolist() == [1, 0, 0], arg

# --- synthetic: dropped constant column; argmax maps back to the ORIGINAL index ---
train3 = np.column_stack([np.full(4, 7.0), train])  # feature 0 is constant
sc3 = fit_scaler(train3)
assert sc3["idx"].tolist() == [1, 2], sc3["idx"]
_, arg3 = score_rows(np.array([[7.0, 3.0, 40.0]]), sc3)
assert arg3.tolist() == [2], arg3  # original index 2, not subset position 1

# --- synthetic: threshold value and strict '>' ---
s = np.arange(1, 1001, dtype=float)
thr = threshold_at(s, 0.99)
assert thr == 991.0, thr
assert alarms(s, thr).sum() == 9  # 992..1000; the value 991 itself does not alarm
assert alarms(s, thr).mean() <= 1 - 0.99

# --- real data, TRAINING rows only: retained counts match the retention table ---
for machine, expected_retained in [("machine-1-1", 30), ("machine-3-11", 25), ("machine-1-2", 32)]:
    tr = load_machine(machine)["train"]
    scaler = fit_scaler(tr)
    assert scaler["idx"].size == expected_retained, (machine, scaler["idx"].size)
    train_scores, _ = score_rows(tr, scaler)
    for q in (0.99, 0.995, 0.999):
        t = threshold_at(train_scores, q)
        frac = alarms(train_scores, t).mean()
        assert frac <= 1 - q + 1e-12, (machine, q, frac)
        print(f"{machine} q={q}: threshold {t:.3f}, train exceedance {frac:.5f} (nominal {1 - q:.5f})")

print("all scoring tests passed")