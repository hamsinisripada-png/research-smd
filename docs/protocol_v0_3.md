# Protocol v0.3 (DRAFT: only data audits and train-only diagnostics have been executed; everything else is proposed)

## 1. Objective
When an anomaly threshold is calibrated on a source machine, how reliably does it transfer to
another machine, and can calibration using the target machine's unlabeled training-score
distribution improve the realized false-positive rate? Empirical investigation; the hypothesis
is not assumed true; novelty is unverified.

## 2. Data and notation
28 machines, 38 features. Per machine: train X_tr (unlabeled), test X_te, point labels y_te from
test_label (0/1). Train and test are separate sequences of different lengths. Original files are
never modified; SHA-256 hashes are recorded and re-checked.

## 3. Conditions
| ID | Scaler + retained features fitted on | Threshold value | Level | Status |
|---|---|---|---|---|
| S1 source-only | source train | q-quantile of source train scores (source scaler) | pair (756) | primary |
| S2a target threshold adaptation | source train (identical to S1) | q-quantile of target train scores, scored with the source scaler | pair (756) | primary |
| S4 target scaler, source threshold value | target train | S1's source threshold value, applied as-is to target scores | pair (756) | primary (approved 2026-10-23) |
| S3 target-only reference | target train | q-quantile of target train scores (target scaler) | target (28) | reference, not a transfer claim |

2x2 view (scaler from / threshold from): S1 = source/source, S2a = source/target,
S4 = target/source, S3 = target/target. S1 vs S2a changes the threshold only (scaler held at the
source). S3 vs S4 changes the threshold value only (scaler held at the target). S4 applies the
source's numeric threshold to scores over the TARGET's retained features; feature-set sizes may
differ, which also changes the score distribution (a max over more or fewer features).
Shared / source-only / target-only feature counts are logged for every S4 result. No pair is
excluded because of feature-set mismatch.
Pooled-source calibration: separate secondary analysis. Random-window evaluation: deferred.
No condition uses target test data for fitting, calibration or feature retention.

## 4. Preprocessing and feature retention
Feature retention rule (approved 2026-10-23). A feature is DROPPED for a fitting machine iff, in
that machine's training data only, (a) it is exactly constant (max == min), or (b) its mode share
p_j = (max over values v of #{t : x_tj = v}) / n_train is >= 0.999 (mode = most frequent exact
value; ties resolved toward the smaller value). Deterministic; no labels; no target test data.
Applied per fitting machine: the retained set is machine-specific, not global. For S1 and S2a it
comes from the source machine's train; for S4 and S3 from the target machine's train. It is never
recomputed from target test data.
Sensitivity analyses (labeled as such, not chances to pick a best rule): A = cutoff 0.99;
B = exact-constant-only removal. The primary analysis uses 0.999.
Rationale for 0.999: at most about 24-29 varying training rows leave an unstable std estimate.
Logged for every machine: retained and dropped features, drop reason, mode value, mode share,
n_train (results/feature_retention_v1.csv).
Rejected: std floor relative to range (arbitrary scale, outlier-dominated); median/MAD scaling
(changes the method; MAD is 0 when more than half the rows share a value).
Mean and std are fitted on the fitting machine's train only, over retained features.

## 5. Score and threshold
z_tj = (x_tj - mu_j) / sigma_j for retained features only. Retained features always have
sigma_j > 0 (asserted in code; no epsilon is added).
Primary score: s_t = max over retained j of |z_tj|. Secondary (predeclared, reported separately):
mean squared z.
Threshold(q) = numpy.quantile(calibration scores, q, method="higher"), q in {0.99, 0.995, 0.999}.
Alarm iff score > threshold (strict). With method="higher" the threshold is an observed
calibration score, so strict exceedance on calibration data can be slightly below 1 - q; the
nominal rate remains 1 - q. It does not guarantee the target-test FPR equals 1 - q.
The grid is fixed before any target-test inspection.

## 6. Outcomes
Primary: realized FPR = FP / (FP + TN) on target-test rows with y = 0. Report raw FPR, nominal
1 - q, and ratio = FPR / (1 - q). Always show raw FPR.
Companion secondary at the same q: recall. Later secondary: pointwise F1, AUPRC, event recall,
false-alarm segments per 1,000 test rows.
Descriptive attribution (preregistered): for every alarm row record the ORIGINAL 0-based feature
index attaining the max |z| (not the position within the retained subset); ties go to the lowest
index (np.argmax). Aggregate per pair, condition and q: alarm count per feature, proportions,
number of unique argmax features, top-5 features; after final evaluation also restricted to
false-alarm rows. Attribution is descriptive, not causal.

## 7. Events (frozen)
- Ground-truth event: maximal contiguous run of 1s in test_label.
- Detected event: contains at least one alarm.
- False-alarm segment: maximal contiguous run of alarms containing no ground-truth anomalous point.
- Event recall = detected events / total events.
- False-alarm segments per 1,000 rows = 1000 * segments / test rows.
Conventions: lenient (an alarm run touching an event is not a false alarm); no smoothing.

## 8. Aggregation and inference
Primary reporting: distribution across target machines (median, IQR, range), per condition and q.
S1 vs S2a and S3 vs S4: paired per-pair differences, summarized per target. Supplementary:
bootstrap over target machines (fixed seed; 28 targets; sources also repeat, so inference is
approximate). No p-values, no row-level binomial intervals (alarms are autocorrelated).
Predeclared stratum: same-group vs cross-group (machine-<group>-<index>). No other strata.
No source-target pair is excluded from the primary analysis based on training-only transfer
diagnostics, test inputs, test labels, anomaly scores, or realized false-positive rates.
Diagnostics are used only for preregistered descriptive reporting and interpretation.

## 9. Leakage rules
Target test inputs and labels are not used for fitting, preprocessing, feature retention,
calibration, hyperparameters, score choice, quantile choice, or model selection. Target test data
is accessed only for final scoring after predictions are fixed. interpretation_label is never
used for modeling; any use is post-hoc and labeled exploratory.
Scoring code is tested on synthetic data and training rows only before any test row is scored.
Disclosure: earlier audits viewed test-input summaries (shift, constants in test); no labels or
model results. Future decisions use train-only diagnostics.

## 10. Status of statements
VERIFIED (executed or from sources):
- 28 machines x 38 features; test rows = label rows on all machines.
- 15 machines have test rows = train rows + 1; README totals differ by 15 (708,405 vs 708,420).
- 17 (machine, feature) pairs constant in train but varying in test: 12 machines, 8 distinct indices.
- 327 label runs (matches an independent paper); anomaly rate 0.42%-15.65% of test rows.
- Label audit: 23/28 machines agree exactly; 5 disagree (1-1, 1-3, 2-3, 3-2, 3-3); 43 points only
  in test_label, 9 only in interpretation ranges.
- Train-only feature diagnostics: 192 exactly constant and 34 near-constant (mode share >= 0.999)
  (machine, feature) pairs; 28 more with mode share in [0.99, 0.999).
- Retention under the primary rule: 838 of 1064 pairs retained (25-32 features per machine).
- Train-only pair diagnostics (756 pairs): 641 pairs have a source-retained feature whose target
  train mean is more than 10 source-stds from the source mean; 250 exceed 100.
ASSUMPTIONS: train half is mostly normal (benchmark convention, unverified); feature index j means
the same thing across machines (not verified for varying features).
HYPOTHESES (untested): extra test row comes from odd-length splitting (secondary source only);
S2a brings realized FPR near nominal except where target train-to-test drift is large; S1 fails
broadly because source scalers mismatch targets (expectation from train-only diagnostics, not a result).
DESIGN CHOICES: everything in sections 3-9.
UNRESOLVED: whether to add a pooled-source analysis later; title wording ("Evaluation-Protocol
Sensitivity") to be revisited once results exist.

## 11. Reproducibility
Fixed config file (tolerance, grid, score, quantile method, seeds); saved CSV outputs; data file
hashes; library versions logged.

## 12. Known limitations
Train contamination could distort scaler and tail thresholds; anomaly prevalence varies; SMD
benchmark criticisms (Wu and Keogh) apply but do not by themselves invalidate results; 28 targets
only; pairs share machines.
Dropping near-constant training features may remove sparse but informative signal and reduce
recall where those features become active on a target. The 0.999 cutoff is a preregistered design
choice, partly arbitrary, examined by sensitivity runs. Features that vary enough in training may
still differ in scale across machines; retention does not guarantee cross-machine score
comparability.
Under S1 and S2a source scalers often mismatch the target, so S1 vs S2a largely measures scaler
mismatch; S4 and S3 isolate threshold-value transfer, but S3 vs S4 also differs in feature-set size.

## 13. Changelog
- v0 (2026-10-23): first draft. SHA-256 D5F0EBF4028DAD71B37D3B2D44B39CE1E009850389291B05981C3704B372E1B9.
- v0.1 (2026-10-23): S2b merged into S3; title kept. SHA-256 22080E457011514868A5EDD769392E0B2ACA3F9F8009314F3DC1678788430FA4.
- v0.2 (2026-10-23): near-constant rule approved. SHA-256 0004A51B8204D0BCABDA22D1AFCB248879A9F13C8F349060D6D78D5554D4E602.
- v0.3 (2026-10-23): condition S4 added; no-pair-exclusion rule; argmax attribution with original
  indices and lowest-index ties; scoring details frozen (no epsilon, strict >).