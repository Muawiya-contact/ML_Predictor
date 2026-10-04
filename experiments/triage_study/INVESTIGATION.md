# Current-data-only four-level investigation

This follow-up keeps the existing 10,000 rows, labels 0/1/2/3, immutable groups,
SapBERT cache and the three paper classifiers. It does not introduce another
dataset, infer replacement labels or drop difficult examples.

First inspect the incumbent's development out-of-fold errors, confidence bins,
multiclass MCC, log loss and Brier score. Exact duplicate input conflicts and
agreement between the three classifiers are screening diagnostics, not evidence
of clinical label correctness. Optional standardized LR coefficients are
associational diagnostics, not causal feature importance.

```bash
python experiments/triage_study/investigate_four_level_errors.py \
  --original output/results/triage_four_level \
  --incumbent output/results/triage_four_level_round4 \
  --peers output/results/triage_learning_detail_audit \
  --bundle triage_model_sapbert \
  --output output/results/triage_error_investigation
```

The bundle must match the specified incumbent. Run this before promoting a new
model, or point `--bundle` to the archived round-four bundle afterward.

Thirteen settings are frozen before scoring: eight LR settings for regularization,
balancing and PCA size, three HGB regularization/leaf settings, and two RF feature
sampling settings. All include the same nineteen original-complaint details.
Scaling and PCA fit only within training folds. All five grouped folds are used
for every setting. No test scores participate in selecting the winner.

```bash
python experiments/triage_study/refine_complaint_details.py \
  --original output/results/triage_four_level \
  --output output/results/triage_detail_refinement
python experiments/triage_study/verify_detail_refinement.py \
  --source output/results/triage_detail_refinement \
  --original output/results/triage_four_level \
  --incumbent output/results/triage_four_level_round4
python experiments/triage_study/finalize_improvement.py \
  --main output/results/triage_four_level_round4 \
  --extra output/results/triage_detail_refinement \
  --original output/results/triage_four_level \
  --output output/results/triage_four_level_round5
python experiments/triage_study/compare_tuned_families.py \
  --source output/results/triage_four_level_round5 \
  --original output/results/triage_four_level
python experiments/triage_study/verify_improvement.py \
  --source output/results/triage_four_level_round5 \
  --original output/results/triage_four_level
```

The combined comparison has 68 settings and 340 full-size CV fits. Selection
retains the original emergency-recall constraint, rather than relaxing it after
each round. Export and live application parity checks remain required before
promotion. Keep the incumbent when no candidate wins the constrained ranking.
Repeated searches reuse development information; test comparisons are retrospective.
Neither 90% nor the older three-level accuracy is guaranteed.

Use fresh output directories for experiment runs. Individual review queues,
OOF arrays and records remain local. Publish only aggregate diagnostics and scores.
The final PDF builder accepts `--investigation output/results/triage_error_investigation`
in addition to `--audit output/results/triage_learning_detail_audit`, so the
investigation remains in the single final report rather than a second PDF.

After verification, export and check the shared application adapter before
replacing the active bundle:

```bash
python experiments/triage_study/export_improved_bundle.py \
  --source output/results/triage_four_level_round5 \
  --original output/results/triage_four_level \
  --incumbent output/results/four_level_round4_bundle \
  --output output/results/four_level_round5_bundle
python experiments/triage_study/verify_improved_serving.py \
  --source output/results/triage_four_level_round5 \
  --original output/results/triage_four_level \
  --bundle output/results/four_level_round5_bundle
```

Preserve the previous bundle locally. Copy the verified artifacts and metrics
into `triage_model_sapbert/`, with `model_manifest.json` copied last, and restart
the GUI. The shared adapter reads the manifest; do not hard-code new scores or
class mappings into individual tabs.

Copy the refinement's `verification.json` into the investigation directory as
`refinement_verification.json`, then regenerate the single report:

```bash
python experiments/triage_study/build_improvement_pdf.py \
  --source output/results/triage_four_level_round5 \
  --original reports/triage_four_level \
  --audit output/results/triage_learning_detail_audit \
  --investigation output/results/triage_error_investigation \
  --output output/pdf/SapBERT_Final_Report.pdf
```

Render and inspect every page before publication. The final comparison includes
all original 768-D/PCA-64 baselines, each tuned classifier, the full development
search and both audits. Probability diagnostics are recomputed from saved
probabilities; lower log loss and Brier score indicate better probability quality.

The later [expanded classifier comparison](EXPANDED_CLASSIFIERS.md) adds the
user-authorized CatBoost, XGBoost and SVM families while preserving these labels
and evaluation partitions. Its report includes the earlier investigation.
