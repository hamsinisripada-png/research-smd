# Protocol v0 (DRAFT: only data audits have been executed; everything else is proposed)

## 1. Objective
When an anomaly threshold is calibrated on a source machine, how reliably does it
transfer to another machine, and can calibration using the target machine's unlabeled
training-score distribution improve the realized false-positive rate? Empirical
investigation; the hypothesis is not assumed true; novelty is unverified.

## 2. Data and notation
28 machines, 38 features. Per machine: train X_tr (unlabeled), test X_te, point labels
y_te from test_label (0/1). Train and test are separate sequences of different lengths.
Original files are never modified; SHA-256 hashes are recorded and re-checked.

## 3. Conditions
| ID | Scaler + retained features fitted on | Threshold calibrated on | Level | Status |
|---|---|---|---|---|
| S1 source-only | source train | source train scores | pair (756) | primary |
| S2a target threshold adaptation | source train (identical to S1) | target train scores, scored with source scaler | pair (756) | primary |
| S2b target preprocessing adaptation | target train | target train scores | target (28) | IDENTICAL to S3 as defined; merge pending approval |
| S3 known-machine reference | target train | target train scores | target (28) | reference, not a transfer claim |
Comparisons: S1 vs S2a changes the threshold only (primary controlled comparison).
S2a vs S3 changes the scaler/feature set (threshold already target-calibrated).
S1 vs S3 changes both. Pooled-source calibration: separate secondary analysis.
Random-window evaluation: deferred. No condition uses target test data for fitting.

## 4. Preprocessing
Per feature: mean and std from the fitting machine's train only. A feature is constant
iff max == min in that train (tolerance 0; matches the audit, where std==0 and max==min
gave identical counts). Constant features are dropped. S1 and S2a apply the identical
source scaler and retained feature set to the target; no refit on target data.
PENDING: rule for near-constant features (tiny nonzero std). To be decided from train-only
diagnostics before any scoring.
Record retained/dropped feature indices per machine and per pair. Limitation: a
source-constant feature that varies on the target cannot contribute to the source-fitted score.

## 5. Score and threshold
Primary score: s(t) = max over retained features j of |z(t,j)|.
Secondary (predeclared, reported separately): mean squared z.
Threshold(q) = numpy.quantile(calibration scores, q, method="higher"), q in {0.99, 0.995, 0.999}.
Alarm iff score > threshold. Nominal calibration exceedance is about 1 - q (ties can lower it;
record the tie fraction). It does not guarantee the target-test FPR equals 1 - q.
The grid is fixed before any target-test inspection.

## 6. Outcomes
Primary: realized FPR = FP / (FP + TN) on target-test rows with y = 0. Report raw FPR,
nominal 1 - q, and ratio = FPR / (1 - q). Always show raw FPR.
Companion secondary at the same q: recall. Later secondary: pointwise F1, AUPRC,
event recall, false-alarm segments per 1,000 test rows.

## 7. Events (frozen)
- Ground-truth event: maximal contiguous run of 1s in test_label.
- Detected event: contains at least one alarm.
- False-alarm segment: maximal contiguous run of alarms containing no ground-truth anomalous point.
- Event recall = detected events / total events.
- False-alarm segments per 1,000 rows = 1000 * segments / test rows.
Conventions: lenient (an alarm run touching an event is not a false alarm); no smoothing.

## 8. Aggregation and inference
Primary reporting: distribution across target machines (median, IQR, range), per condition
and q. S1 vs S2a: paired per-pair differences, summarized per target. Supplementary:
bootstrap over target machines (fixed seed; 28 targets; sources also repeat, so inference
is approximate). No p-values, no row-level binomial intervals (alarms are autocorrelated).
Predeclared stratum: same-group vs cross-group (machine-<group>-<index>). No other strata.

## 9. Leakage rules
Target test inputs and labels are not used for fitting, preprocessing, calibration,
feature choice, hyperparameters, score choice, quantile choice, or model selection.
Target test data is accessed only for final scoring after predictions are fixed.
interpretation_label is never used for modeling; any use is post-hoc and labeled exploratory.
Disclosure: earlier audits viewed test-input summaries (shift, constants in test); no labels
or model results. Future decisions use train-only diagnostics.

## 10. Status of statements
VERIFIED (executed or from sources):
- 28 machines x 38 features; test rows = label rows on all machines.
- 15 machines have test rows = train rows + 1; README totals differ by 15 (708,405 vs 708,420).
- 17 (machine, feature) pairs constant in train but varying in test: 12 machines, 8 distinct indices.
- 327 label runs (matches an independent paper); anomaly rate 0.42%-15.65% of test rows.
- Label audit: 23/28 machines agree exactly; 5 disagree (1-1, 1-3, 2-3, 3-2, 3-3); 43 points only in
  test_label, 9 only in interpretation ranges.
ASSUMPTIONS: train half is mostly normal (benchmark convention, unverified); feature index j
means the same thing across machines (not verified for varying features).
HYPOTHESES (untested): extra test row comes from odd-length splitting (secondary source only);
S2a brings realized FPR near nominal except where target train-to-test drift is large.
DESIGN CHOICES: everything in sections 3-9.
UNRESOLVED: S2b merge; near-constant rule; title scope; whether to add pooled-source analysis later.

## 11. Reproducibility
Fixed config file (tolerance, grid, score, quantile method, seeds); saved CSV outputs;
data file hashes; library versions logged.

## 12. Known limitations
Train contamination could distort scaler and tail thresholds; anomaly prevalence varies;
SMD benchmark criticisms (Wu and Keogh) apply but do not by themselves invalidate results;
28 targets only; pairs share machines.