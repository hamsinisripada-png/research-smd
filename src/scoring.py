"""scoring.py -- step 7: scaler fitting, max-|z| scores, thresholds, alarms.
Pure functions: they read no files. They see test data only if a caller passes it in."""
import numpy as np

from feature_retention import mode_info

PRIMARY_CUTOFF = 0.999


def retained_indices(train, cutoff=PRIMARY_CUTOFF):
    """ORIGINAL 0-based indices of retained features (approved rule; training data only).
    cutoff=None means exact-constant-only removal (sensitivity B)."""
    n = train.shape[0]
    keep = []
    for j in range(train.shape[1]):
        col = train[:, j]
        if col.max() == col.min():
            continue
        if cutoff is not None:
            _, mode_count = mode_info(col)
            if mode_count / n >= cutoff:
                continue
        keep.append(j)
    return np.array(keep, dtype=int)


def fit_scaler(train, cutoff=PRIMARY_CUTOFF):
    """Fit retained set, means and stds on a fitting machine's training data only."""
    idx = retained_indices(train, cutoff)
    assert idx.size > 0, "no retained features"
    mean = train[:, idx].mean(axis=0)
    std = train[:, idx].std(axis=0)
    assert (std > 0).all(), "a retained feature has zero std"
    return {"idx": idx, "mean": mean, "std": std}


def score_rows(X, scaler):
    """Max |z| per row over retained features, and the ORIGINAL index attaining it."""
    z = np.abs((X[:, scaler["idx"]] - scaler["mean"]) / scaler["std"])
    pos = z.argmax(axis=1)  # position inside the retained subset; first maximum wins
    return z.max(axis=1), scaler["idx"][pos]


def threshold_at(calibration_scores, q):
    """q-quantile of calibration scores (an observed score; method='higher')."""
    return float(np.quantile(calibration_scores, q, method="higher"))


def alarms(scores, threshold):
    """Alarm iff the score is strictly greater than the threshold."""
    return scores > threshold