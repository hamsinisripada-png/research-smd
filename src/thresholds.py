"""thresholds.py -- thresholds and TRAIN-ONLY exceedances for S1, S2a, S4 (pairs) and S3 (targets),
for one retention rule. No test row is read.
Usage: python src\\thresholds.py [primary_0.999 | sensA_0.99 | sensB_exact_only]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data_validation import load_machine, machine_names, sha256_of_file
from scoring import alarms, fit_scaler, score_rows, threshold_at

QUANTILES = (0.99, 0.995, 0.999)
N_MACHINES = 28
PRIMARY = "primary_0.999"
RULES = {PRIMARY: 0.999, "sensA_0.99": 0.99, "sensB_exact_only": None}


def make_row(condition, source, target, q, thr, calib, target_train, counts):
    return {
        "condition": condition, "source": source, "target": target, "q": q,
        "threshold": thr, "nominal_exceedance": 1 - q,
        "calib_exceedance_realized": calib, "target_train_exceedance": target_train,
        **counts,
    }


if __name__ == "__main__":
    rule = sys.argv[1] if len(sys.argv) > 1 else PRIMARY
    assert rule in RULES, f"unknown rule {rule!r}; choose from {list(RULES)}"
    cutoff = RULES[rule]
    print("retention rule:", rule)

    out_dir = Path(__file__).resolve().parent.parent / "results"
    names = machine_names()
    train = {m: load_machine(m)["train"] for m in names}  # about 250 MB in memory
    scaler = {m: fit_scaler(train[m], cutoff) for m in names}
    sizes = [scaler[m]["idx"].size for m in names]
    print("retained features per machine, min / max:", min(sizes), "/", max(sizes))
    own = {m: score_rows(train[m], scaler[m])[0] for m in names}  # each machine, own scaler
    self_thr = {(m, q): threshold_at(own[m], q) for m in names for q in QUANTILES}

    if rule == PRIMARY:  # step 8 table exists for the primary rule only
        cal = pd.read_csv(out_dir / "calibration_v1.csv")
        for _, r in cal.iterrows():
            assert abs(self_thr[(r["machine"], r["q"])] - r["threshold"]) < 1e-9, r["machine"]
        print("self-thresholds match calibration_v1.csv: OK")

    rows = []
    for t in names:  # S3: one row per target and q
        counts = {"n_retained_source": np.nan, "n_retained_target": int(scaler[t]["idx"].size),
                  "n_shared": np.nan, "n_source_only": np.nan, "n_target_only": np.nan}
        for q in QUANTILES:
            thr = self_thr[(t, q)]
            calib = float(alarms(own[t], thr).mean())
            rows.append(make_row("S3", "", t, q, thr, calib, calib, counts))

    for s in names:
        Ks = set(scaler[s]["idx"].tolist())
        for t in names:
            if s == t:
                continue
            Kt = set(scaler[t]["idx"].tolist())
            counts = {"n_retained_source": len(Ks), "n_retained_target": len(Kt),
                      "n_shared": len(Ks & Kt), "n_source_only": len(Ks - Kt),
                      "n_target_only": len(Kt - Ks)}
            t_under_s = score_rows(train[t], scaler[s])[0]  # target train, SOURCE scaler
            t_under_s_again = score_rows(train[t], scaler[s])[0]  # independent second scoring
            for q in QUANTILES:
                thr_src = self_thr[(s, q)]
                calib_src = float(alarms(own[s], thr_src).mean())
                rows.append(make_row("S1", s, t, q, thr_src, calib_src,
                                     float(alarms(t_under_s, thr_src).mean()), counts))
                thr_2a = threshold_at(t_under_s, q)
                calib_2a = float(alarms(t_under_s, thr_2a).mean())
                tt_2a = float(alarms(t_under_s_again, thr_2a).mean())
                assert calib_2a == tt_2a, (s, t, q)  # equal by construction
                rows.append(make_row("S2a", s, t, q, thr_2a, calib_2a, tt_2a, counts))
                rows.append(make_row("S4", s, t, q, thr_src, calib_src,
                                     float(alarms(own[t], thr_src).mean()), counts))

    df = pd.DataFrame(rows)
    expected_rows = len(QUANTILES) * (3 * N_MACHINES * (N_MACHINES - 1) + N_MACHINES)
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"

    name = "thresholds_v1.csv" if rule == PRIMARY else f"thresholds_{rule}_v1.csv"
    out_path = out_dir / name
    df.to_csv(out_path, index=False)

    print("rows:", len(df))
    print(df.groupby("condition").size().to_string())
    print("S2a / S3 consistency (target_train == calibration exceedance): OK")
    print("saved:", out_path)
    print("SHA-256:", sha256_of_file(out_path))