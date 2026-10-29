"""pair_diagnostics.py -- step 6: train-only scale-mismatch diagnostics for all 756 ordered pairs.
Uses TRAINING data of both machines only (no test data, no labels). Descriptive:
all rules are frozen and no pair is excluded because of these numbers."""
from pathlib import Path

import numpy as np
import pandas as pd

from data_validation import load_machine, machine_names
from feature_retention import mode_info

PRIMARY_CUTOFF = 0.999
N_MACHINES = 28


def machine_stats(train):
    """Per-feature mean, std, and the primary-rule retained mask (train only)."""
    n = train.shape[0]
    mean, std = train.mean(axis=0), train.std(axis=0)
    retained = np.zeros(train.shape[1], dtype=bool)
    for j in range(train.shape[1]):
        is_const = train[:, j].max() == train[:, j].min()
        _, mode_count = mode_info(train[:, j])
        retained[j] = (not is_const) and (mode_count / n < PRIMARY_CUTOFF)
    return {"mean": mean, "std": std, "retained": retained}


if __name__ == "__main__":
    names = machine_names()
    stats = {m: machine_stats(load_machine(m)["train"]) for m in names}

    rows = []
    for s in names:
        for t in names:
            if s == t:
                continue
            S, T = stats[s], stats[t]
            keep = S["retained"]  # the SOURCE decides which features are used
            sigma_s = S["std"][keep]  # > 0 for every retained feature
            ratio = T["std"][keep] / sigma_s
            shift_z = np.abs(T["mean"][keep] - S["mean"][keep]) / sigma_s
            rows.append({
                "source": s,
                "target": t,
                "same_group": s.split("-")[1] == t.split("-")[1],
                "n_retained_source": int(keep.sum()),
                "max_std_ratio": float(ratio.max()),
                "n_std_ratio_gt_10": int((ratio > 10).sum()),
                "max_mean_shift_z": float(shift_z.max()),
                "n_mean_shift_z_gt_10": int((shift_z > 10).sum()),
            })

    df = pd.DataFrame(rows)
    assert len(df) == N_MACHINES * (N_MACHINES - 1), f"expected 756 pairs, got {len(df)}"

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_path = out_dir / "pair_diagnostics_v1.csv"
    df.to_csv(out_path, index=False)

    # Cross-check against the retention table
    ret = pd.read_csv(out_dir / "feature_retention_v1.csv")
    ret = ret[ret["rule"] == "primary_0.999"].groupby("machine")["retained"].sum()
    per_source = df.groupby("source")["n_retained_source"]
    assert per_source.nunique().eq(1).all(), "retained count varies within a source"
    assert (ret.reindex(per_source.first().index) == per_source.first()).all(), \
        "retained counts disagree with feature_retention_v1.csv"
    print("pairs:", len(df))
    print("cross-check with feature_retention_v1.csv: OK")

    pd.set_option("display.width", 200)
    for col in ["max_std_ratio", "max_mean_shift_z"]:
        q = df[col].quantile([0.5, 0.9]).tolist() + [df[col].max()]
        print(f"{col}: median {q[0]:.2f} / 90th pct {q[1]:.2f} / max {q[2]:.2f}")
    print("pairs with max_mean_shift_z > 10:", int((df["max_mean_shift_z"] > 10).sum()))
    print("pairs with max_mean_shift_z > 100:", int((df["max_mean_shift_z"] > 100).sum()))
    print("\nmedian max_mean_shift_z, same-group vs cross-group:")
    print(df.groupby("same_group")["max_mean_shift_z"].median().to_string())
    print("\n5 sources with the largest median max_mean_shift_z:")
    print(df.groupby("source")["max_mean_shift_z"].median().nlargest(5).to_string())
    print("\n10 pairs with the largest max_mean_shift_z:")
    print(df.nlargest(10, "max_mean_shift_z")[
        ["source", "target", "max_std_ratio", "max_mean_shift_z"]].to_string(index=False))
    print("\nsaved:", out_path)