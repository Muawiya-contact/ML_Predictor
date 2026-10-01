# Current four-level SapBERT comparison

Selected model: frozen SapBERT PCA-128 + 50 patient features + 19 original-complaint
features, balanced Logistic Regression C=10 (197 inputs). All 10,000 source rows,
labels 0/1/2/3 and the original grouped partitions are unchanged.

The combined search contains 68 settings and 340 full-size grouped-fold fits.
Development mean macro F1 is 89.65%, versus 89.38% immediately before this
refinement. Emergency recall is 94.06%. The paired development F1 gain interval
(-0.08 to +0.64 percentage points) includes zero and excludes selection uncertainty;
a reliable incremental gain is not established.

On the previously examined 1,999 test rows: accuracy 89.29%, macro precision
89.67%, macro recall 89.81%, macro F1 89.73%, emergency recall 95.10%, and
under-triage 4.65%. These are retrospective comparisons, not independent clinical
validation. The three tuned-family results and all original full-768/PCA-64
comparisons remain in the report. Saved concepts feed SapBERT; original complaints
provide detail features. Live translation accuracy was not evaluated.

Use [SapBERT_Final_Report.pdf](SapBERT_Final_Report.pdf) as the single current
report. Its eleven pages include both audits, every development setting, original
baselines, tuned classifiers, matrices, class scores and literature context.
All pages were rendered and visually inspected. The shared application adapter
reproduces all 1,999 predictions and probabilities; 13 live embeddings and a
live English prediction passed verification.

See [the investigation](../triage_error_investigation/) and
[reproduction commands](../../experiments/triage_study/INVESTIGATION.md).
Individual records, OOF arrays and review queues remain local.

Validation: 42 targeted unit tests and the real six-tab GUI audit passed.

The [previous-versus-latest comparison](../triage_historical_comparison/) explains the changed labels, matrix counts and score differences in a separate companion PDF.
