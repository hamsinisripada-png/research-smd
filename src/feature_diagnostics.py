"""feature_diagnostics.py -- step 4: per-feature, TRAIN-ONLY diagnostics for every machine.
Read-only. Uses no test data and no labels, so results may inform design decisions."""
from pathlib import Path

import numpy as np
import pandas as pd

from data_validation import (
    NEAR_CONST_MODE_FRACTION,
    load_machine,
    machine_names,
    mode_fraction,
)

N_MACHINES = 28
N_FEATURES = 38


def feature_rows(machine, train):
    """One dict per feature for one machine, using the training half only."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    rows = []
    for j in range(train.shape[1]):
        col = train[:, j]
        lo, hi = col.min(), col.max()
        is_const = bool(lo == hi)
        mf = mode_fraction(col)
        if is_const:
            max_z = frac_z3 = std_over_range = float("nan")
        else:
            z = np.abs((col - mean[j]) / std[j])
            max_z = float(z.max())
            frac_z3 = float((z > 3).mean())
            std_over_range = float(std[j] / (hi - lo))
        rows.append({
            "machine": machine,
            "feature": j,  # 0-based
            "train_mean": float(mean[j]),
            "train_std": float(std[j]),
            "train_min": float(lo),
            "train_max": float(hi),
            "train_range": float(hi - lo),
            "std_over_range": std_over_range,
            "n_unique": int(np.unique(col).size),
            "mode_frac": float(mf),
            "is_const": is_const,
            "is_near_const": (not is_const) and mf >= NEAR_CONST_MODE_FRACTION,
            "max_abs_z_train": max_z,
            "frac_abs_z_gt3_train": frac_z3,
        })
    return rows


if __name__ == "__main__":
    all_rows = []
    for machine in machine_names():
        data = load_machine(machine)
        all_rows.extend(feature_rows(machine, data["train"]))
    df = pd.DataFrame(all_rows)
    assert len(df) == N_MACHINES * N_FEATURES, f"expected {N_MACHINES * N_FEATURES} rows, got {len(df)}"

    out_path = Path(__file__).resolve().parent.parent / "results" / "feature_diagnostics_v1.csv"
    df.to_csv(out_path, index=False)

    pd.set_option("display.width", 220)
    cols = ["machine", "feature", "mode_frac", "n_unique", "train_std",
            "std_over_range", "max_abs_z_train"]

    print("rows:", len(df))
    print("exactly constant (machine, feature) pairs:", int(df["is_const"].sum()))
    print("near-constant pairs (mode_frac >= %s):" % NEAR_CONST_MODE_FRACTION,
          int(df["is_near_const"].sum()))
    band = df[(~df["is_const"]) & (df["mode_frac"] >= 0.99) & (df["mode_frac"] < NEAR_CONST_MODE_FRACTION)]
    print("pairs with mode_frac in [0.99, 0.999) (just below the cutoff):", len(band))

    print("\n=== all near-constant pairs ===")
    print(df.loc[df["is_near_const"], cols].to_string(index=False))

    rest = df[(~df["is_const"]) & (~df["is_near_const"])]
    print("\n=== 10 smallest std_over_range among the other non-constant pairs ===")
    print(rest.nsmallest(10, "std_over_range")[cols].to_string(index=False))

    print("\n=== 10 largest max_abs_z_train (all non-constant pairs) ===")
    print(df[~df["is_const"]].nlargest(10, "max_abs_z_train")[cols].to_string(index=False))

    print("\nsaved:", out_path)