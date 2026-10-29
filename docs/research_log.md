## 2026-10-22 — SMD audit v1
FACT: 28 machines, 38 features each; no non-finite values; labels binary; pooled anomaly rate 4.16%.
FACT: 15 machines have test_rows = train_rows + 1; verified present in raw files (not a parsing artifact). Cause unknown.
FACT: ~327 anomaly events total; machine-2-8 has 1 event; machine-1-6's longest segment holds ~85% of its anomaly points.
FACT: train->test mean shift ranges from <1 to 260.8 train-stds (machine-3-4); cause (drift vs anomalies) not separated.
OPEN: constant-feature handling; label temporal alignment (interpretation_label cross-check).
DESIGN NOTE: aggregate per machine; report event counts alongside event metrics.

## 2026-10-22 — feature audit
FACT: 175/1064 machine-feature pairs constant in both halves; 17 constant in train only (all 0.0 in train; <=1.83% of test rows differ; 3 differ on a single row); 3 constant in test only.
FACT: feature 7 constant on all 28 machines; features 26, 36, 37 constant on 26-27.
DESIGN CHOICE: drop features constant in the fitting data (label-free); log the protocol-dependent feature-set confound; ablate later.
ASSUMPTION: feature index i plays a similar role across machines (not verified for varying features).

## 2026-10-22 — alignment check (machine-1-1)
FACT: interpretation_label ranges = [start:end) 0-based, exclusive end; 7/8 segments identical to test_label runs.
FACT: segment 1: test_label 15849..16395 vs interpretation 15849-16368 (27 rows longer in test_label). Cause unknown.
DESIGN CHOICE: test_label is the evaluation ground truth; no label editing; disagreements reported.
OBSERVATION: three events of length 2-3 rows on this machine; event lengths overall range 2..3161.
OPEN: do other machines show disagreements?

## 2026-10-22 — documentation check
VERIFIED: README table lists train 708,405 vs test 708,420 (diff 15 = machines with extra test row); 2026 paper reports odd-length series give the test set the extra point.
VERIFIED: machine-1-1/1-2 segment stats match the independent 2026 paper; full dataset 327 segments there.
VERIFIED: data_preprocess.py for SMD loads train/test/test_label only; README gives no index convention for interpretation_label.
UNRESOLVED: 27-row disagreement on machine-1-1 segment 1; other machines pending.

## 2026-10-23 — all-machine interpretation audit v1
FACT: total label runs = 327 (matches independent paper). 25/28 machines have equal run/range counts.
FACT: counts differ on machine-1-3 (12 vs 11), machine-3-2 (10 vs 9), machine-3-3 (26 vs 28).
FACT: v1 order-based pairing flagged 8 segments; 6 (machines 1-4, 1-8, 2-1) are identical intervals in swapped line order -> v1 flaw (assumed chronological order), not a data issue.
FACT: real boundary differences: machine-1-1 seg0 (+27 rows in test_label), machine-2-3 seg3 (-9 rows).
HYPOTHESIS: unequal counts come from touching/overlapping ranges merging into one label run (untested).
DESIGN CHOICE: events = maximal runs in test_label; note machines where interpretation_label segments differently.

## 2026-10-23 — protocol v0.1
DECISION (approved): S2b merged into S3 (identical as defined); S3 = target-only reference, 28 results.
DECISION (approved): keep title for now; revisit "Evaluation-Protocol Sensitivity" wording once results exist.
FILE: protocol_v0.md saved, 6086 bytes, SHA-256 D5F0EBF4...E1B9 (kept unchanged as audit trail); protocol_v0_1.md created.
UNRESOLVED: near-constant-feature rule (needs train-only diagnostics); pooled-source secondary analysis timing.

## 2026-10-23 — protocol v0.1 verified
FILE: docs\protocol_v0_1.md, 6266 bytes, SHA-256 22080E457011514868A5EDD769392E0B2ACA3F9F8009314F3DC1678788430FA4
CHECK: S2b appears only in the merged S3 row and the changelog; heading reads v0.1.
FILE: docs\protocol_v0.md kept unchanged (SHA-256 D5F0EBF4028DAD71B37D3B2D44B39CE1E009850389291B05981C3704B372E1B9).
DATA: machine-1-1 fingerprints from data_validation.py step 1 (train ecebd293..., test bdb801d8..., label cddf3344...).

## 2026-10-23 — data_validation.py step 2
CHECK: machine-1-1 audit row reproduced earlier audits (rows, constants, events, hashes).
NEW: min_nonzero_train_std 1.47e-05 on machine-1-1; not interpretable alone (needs feature range and mode fraction).
NEXT: step 3 all-machine CSV; step 4 per-feature train-only diagnostics for the near-constant rule.

## 2026-10-23 — step 3 all-machine audit
CHECK: 28 audited, 0 failing, 327 events, pooled rate 0.0416, 15 machines +1 test row, 17 train-const/test-varying pairs; exit code 0.
FILE: results\data_audit_v1.csv (SHA-256: paste full hash).
FINDING: near-constant train features exist (machine-1-2: 1, machine-1-3: 3 at mode_frac >= 0.999).
NEXT: step 4 feature_diagnostics.py (train-only) to decide near-constant rule before any scoring.

## 2026-10-23 — step 4 feature diagnostics (train-only)
CHECK: 1064 rows; 192 exactly constant; 34 near-constant (matches data_audit_v1.csv sum).
FINDING: near-constant features are mostly 1-row deviations; max_abs_z_train saturates at sqrt(n-1) (verified on 3 rows), so not a decision criterion.
FINDING: std_over_range does not separate degenerate from healthy features; range-based std floor rejected.
FINDING: 28 more pairs lie in mode_frac [0.99, 0.999); cutoff-sensitive.
OBSERVATION: near-constant indices recur (16 x7, 17 x6, 28 x5, 32 x5).
PROPOSED (pending approval): drop features with >= 99.9% rows sharing one value (plus exact constants); sensitivity runs at 0.99 and exact-constant-only.
FILE: results\feature_diagnostics_v1.csv; results\data_audit_v1.csv (paste full hash: (Get-FileHash results\data_audit_v1.csv).Hash).

## 2026-10-23 — near-constant rule approved
DECISION: drop feature if exactly constant or mode share >= 0.999, per fitting machine, training data only; same rule in S1, S2a, S3.
DECISION: sensitivity A (0.99) and B (exact-constant-only) preregistered; not chances to pick a best rule.
FILE: docs\protocol_v0_2.md (SHA-256: paste).
NEXT: feature_retention.py output check; then pair-level train-only scale-mismatch diagnostic; then scoring code.

## 2026-10-23 — step 5 retention table, step 6 pair diagnostics
CHECK: feature_retention.py: 3192 rows; dropped 226 / 254 / 192 (primary / sens A / exact-only); cross-check OK.
FILE: docs\protocol_v0_2.md SHA-256 0004A51B8204D0BCABDA22D1AFCB248879A9F13C8F349060D6D78D5554D4E602.
PRE-COMMITMENT (pending approval): no pair excluded because of train-only pair diagnostics.
NEXT: pair_diagnostics.py output; then scoring functions tested on synthetic data and training data only.

## 2026-10-23 — step 6 pair diagnostics (train-only)
CHECK: 756 pairs; cross-check with retention table OK.
FINDING: 641/756 pairs have a source-retained feature with target train mean > 10 source-stds away; 250 > 100; max 34186 (machine-1-1 -> machine-3-5).
FINDING: fragile sources by median shift: machine-1-1 (2032), 2-8 (962), 3-9 (893), 3-1 (851), 2-6 (289).
EXPECTATION (not a result): S1 will mostly fail for scaler-mismatch reasons; S2a will recover near-nominal FPR largely by construction.
HYPOTHESIS: machine-1-1's tiny-std retained feature (1.47e-05) drives its fragility (unverified).
PENDING APPROVAL: no pair exclusions; add S4 (target scaler + source threshold value); record argmax-feature alarm counts.
FILE: results\pair_diagnostics_v1.csv.

## 2026-10-23 — protocol v0.3, scoring functions
DECISIONS (approved): no pair exclusions; add S4 (target scaler, source threshold value); argmax-feature counts with lowest-index ties.
FIXES TO PASTED WORDING: argmax logs ORIGINAL feature index; no epsilon (assert sigma > 0); S3 vs S4 feature-count confound logged as limitation.
FILE: docs\protocol_v0_3.md (SHA-256: paste); v0-v0.2 kept unchanged.
NEXT: run tests\test_scoring.py; then pair-level scoring on TRAINING rows only (calibration distributions) before any test row is scored.
## 2026-10-24 — step 7 scoring tests; step 8 calibration
CHECK: tests\test_scoring.py passed (synthetic + train-only): known z, original-index argmax, lowest-index ties, strict >, retained counts 30/25/32.
FINDING: thresholds differ across machines at the same q (q=0.99: 7.2, 4.1, 15.3 on three machines); preview only.
FINDING: calibration exceedance below nominal (machine-1-2 q=0.995: 0.00291 vs 0.005); cause hypothesised as score ties.
PENDING APPROVAL: report realized calibration exceedance beside nominal (protocol v0.4).
OUTSTANDING: protocol_v0_3.md verification output (size, S4 lines, hash).
NEXT: calibration.py output; S2a pair thresholds (train only); freeze threshold files and hash them before any test row is scored.

## 2026-10-24 — protocol v0.4
DECISION (approved): report nominal and realized calibration exceedance beside raw target FPR (primary).
FIXES: calibration scores defined per condition; zero-denominator ratios reported as missing; tie cause stays a hypothesis.
PENDING APPROVAL: optional target_train_exceedance column.
FILE: docs\protocol_v0_4.md (SHA-256: paste).
OUTSTANDING: protocol_v0_3.md verification; calibration.py output.
## 2026-10-24 — target_train_exceedance
DECISION (approved): add target_train_exceedance to the v0.4 reporting columns (descriptive; S2a/S3 consistency check).
FIX: wording says "assumed-mostly-normal" training data; column order set.
FILE: docs\protocol_v0_4.md (SHA-256: paste).
OUTSTANDING: calibration.py output; protocol_v0_3.md verification.

## 2026-10-24 — step 8 calibration table; protocol v0.4
FINDING: z-unit thresholds differ 3-4.6x across machines at the same q (q=0.99: 3.30-15.32).
FINDING: tie fraction at threshold up to 0.0112 (q=0.99); exceedance + tie fraction >= nominal on all 5 worst rows.
FILES: results\calibration_v1.csv; docs\protocol_v0_4.md SHA-256 80016066738B2FDC00F155C9447095C83499A688108CE7C7444C69888EC140E3.
CORRECTION: earlier I said two Select-String matches for target_train_exceedance; one is correct (Edit B uses different wording).
NEXT: thresholds.py output; freeze hashes of calibration_v1.csv and thresholds_v1.csv; then evaluation runner tested on synthetic data.