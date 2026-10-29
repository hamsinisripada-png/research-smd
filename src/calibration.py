"""calibration.py -- step 8: self-calibration table (each machine scored by its OWN scaler).
TRAINING data only. Feeds S1 and S4 (source threshold value) and S3 (target threshold)."""
from pathlib import Path

import pandas as pd

from data_validation import load_machine, machine_names
from scoring import alarms, fit_scaler, score_rows, threshold_at

QUANTILES = (0.99, 0.995, 0.999)
N_MACHINES = 28

# Values already printed by tests\test_scoring.py (3 decimals) -> regression check
EXPECTED = {
    ("machine-1-1", 0.99): "7.214", ("machine-1-1", 0.995): "9.473", ("machine-1-1", 0.999): "19.332",
    ("machine-3-11", 0.99): "4.102", ("machine-3-11", 0.995): "6.515", ("machine-3-11", 0.999): "15.090",
    ("machine-1-2", 0.99): "15.319", ("machine-1-2", 0.995): "19.367", ("machine-1-2", 0.999): "30.709",
}

if __name__ == "__main__":
    rows = []
    for machine in machine_names():
        train = load_machine(machine)["train"]
        scaler = fit_scaler(train)  # primary retention rule
        scores, _ = score_rows(train, scaler)
        for q in QUANTILES:
            thr = threshold_at(scores, q)
            calib = float(alarms(scores, thr).mean())
            rows.append({
                "machine": machine,
                "q": q,
                "n_train": train.shape[0],
                "n_retained": int(scaler["idx"].size),
                "threshold": thr,
                "nominal_exceedance": 1 - q,
                "calib_exceedance": calib,
                "calib_over_nominal": calib / (1 - q),
                "tie_fraction_at_threshold": float((scores == thr).mean()),
            })

    df = pd.DataFrame(rows)
    assert len(df) == N_MACHINES * len(QUANTILES), f"unexpected row count {len(df)}"

    for (machine, q), expected in EXPECTED.items():
        got = df[(df["machine"] == machine) & (df["q"] == q)]["threshold"].iloc[0]
        assert f"{got:.3f}" == expected, (machine, q, got, expected)
    print("regression check against test_scoring.py output: OK")

    out_path = Path(__file__).resolve().parent.parent / "results" / "calibration_v1.csv"
    df.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    print("rows:", len(df))
    for q, g in df.groupby("q"):
        print(f"q={q}: threshold min/median/max = {g.threshold.min():.2f} / "
              f"{g.threshold.median():.2f} / {g.threshold.max():.2f}; "
              f"calib/nominal min/median = {g.calib_over_nominal.min():.2f} / "
              f"{g.calib_over_nominal.median():.2f}; "
              f"max tie fraction at threshold = {g.tie_fraction_at_threshold.max():.4f}")
    print("\n5 rows with the smallest calib_over_nominal:")
    print(df.nsmallest(5, "calib_over_nominal")[
        ["machine", "q", "threshold", "calib_exceedance", "tie_fraction_at_threshold"]
    ].to_string(index=False))
    print("\nsaved:", out_path)