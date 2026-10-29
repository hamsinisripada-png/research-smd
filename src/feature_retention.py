"""feature_retention.py -- step 5: apply the approved feature-retention rule per machine.
Uses TRAINING data only. Read-only on the dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

from data_validation import load_machine, machine_names

N_MACHINES = 28
N_FEATURES = 38
RULES = {
    "primary_0.999": 0.999,
    "sensA_0.99": 0.99,
    "sensB_exact_only": None,  # None = drop exact constants only
}


def mode_info(column):
    """Most frequent exact value and its count (ties go to the smaller value)."""
    values, counts = np.unique(column, return_counts=True)
    k = counts.argmax()
    return float(values[k]), int(counts[k])


def retention_rows(machine, train, rule_name, cutoff):
    """One dict per feature: kept or dropped, and why."""
    n = train.shape[0]
    rows = []
    for j in range(train.shape[1]):
        mode_value, mode_count = mode_info(train[:, j])
        share = mode_count / n
        is_const = bool(train[:, j].max() == train[:, j].min())
        if is_const:
            retained, reason = False, "exact_constant"
        elif cutoff is not None and share >= cutoff:
            retained, reason = False, f"mode_share>={cutoff}"
        else:
            retained, reason = True, ""
        rows.append({
            "rule": rule_name,
            "machine": machine,
            "feature": j,  # 0-based
            "retained": retained,
            "drop_reason": reason,
            "mode_value": mode_value,
            "mode_share": share,
            "n_train": n,
        })
    return rows


if __name__ == "__main__":
    all_rows = []
    for machine in machine_names():
        train = load_machine(machine)["train"]
        for rule_name, cutoff in RULES.items():
            rows = retention_rows(machine, train, rule_name, cutoff)
            assert any(r["retained"] for r in rows), f"{machine} {rule_name}: no features retained"
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    assert len(df) == N_MACHINES * N_FEATURES * len(RULES), f"unexpected row count {len(df)}"

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_path = out_dir / "feature_retention_v1.csv"
    df.to_csv(out_path, index=False)

    print("rows:", len(df))
    summary = df.groupby("rule")["retained"].agg(retained="sum", total="count")
    summary["dropped"] = summary["total"] - summary["retained"]
    print(summary.to_string())

    per_machine = df.pivot_table(index="machine", columns="rule", values="retained", aggfunc="sum")
    per_machine = per_machine.reindex(machine_names())[list(RULES)].astype(int)
    print("\nretained features per machine:")
    print(per_machine.to_string())

    # Cross-check against the separately written diagnostics file
    diag = pd.read_csv(out_dir / "feature_diagnostics_v1.csv")
    prim = df[df["rule"] == "primary_0.999"][["machine", "feature", "retained"]]
    merged = prim.merge(diag[["machine", "feature", "is_const", "is_near_const"]],
                        on=["machine", "feature"])
    assert len(merged) == N_MACHINES * N_FEATURES, "merge lost rows"
    expected_dropped = merged["is_const"] | merged["is_near_const"]
    assert (expected_dropped == ~merged["retained"]).all(), \
        "primary rule disagrees with feature_diagnostics_v1.csv"
    print("\ncross-check with feature_diagnostics_v1.csv: OK")
    print("saved:", out_path)