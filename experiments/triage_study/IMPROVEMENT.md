# Four-level improvement round

This round preserves all source labels, SapBERT settings and original group
partitions. It searches 29 predeclared configurations over the five development
folds (145 fits). A maximum of three separate fold processes run concurrently;
preprocessing is fitted independently inside each fold.

Candidates include Logistic Regression with C=10/100/1000, balanced/unbalanced
training and PCA-64/128/256; whitened PCA with C=0.1; regularized HGB with 7/15
leaves; and Random Forest. Structured-only models are controls. SapBERT remains
frozen. No new features, labels or clinical facts are invented.

Selection uses highest mean development macro F1 subject to emergency recall
being within one percentage point of the incumbent. Accuracy breaks ties. This
is an experimental selection criterion, not a clinical safety guarantee. The
incumbent is reproduced on the same folds. All candidates are reported.

The old test set was already examined. New results on it are **retrospective**,
not an untouched validation claim. Selection is frozen before these scores are
computed. Fresh, independently labelled records are needed for confirmation.
The OOF paired group bootstrap is conditional on the selected predictions; it
does not account for selection optimism from the expanded search.

Run from the repository root with the pinned project environment:

```bash
.venv/bin/python experiments/triage_study/improve_four_level.py \
  --source output/results/triage_four_level \
  --output output/results/triage_four_level_improved_parallel
.venv/bin/python experiments/triage_study/verify_improvement.py \
  --source output/results/triage_four_level_improved_parallel \
  --original output/results/triage_four_level
.venv/bin/python experiments/triage_study/export_improved_bundle.py \
  --source output/results/triage_four_level_improved_parallel \
  --original output/results/triage_four_level \
  --incumbent triage_model_sapbert \
  --output output/results/triage_four_level_improved_parallel/exported_bundle
.venv/bin/python experiments/triage_study/build_improvement_pdf.py \
  --source output/results/triage_four_level_improved_parallel \
  --original output/results/triage_four_level \
  --output output/pdf/SapBERT_Four_Level_Improved_Comparison.pdf
```

Use a fresh output directory for every run. The verification script checks
input hashes, complete folds, baseline reproduction, rank, class metrics,
confusion matrices and all row identities. Bundle export refuses to silently
substitute a structured-only control for the SapBERT GUI. Promotion is separate
from export and requires live-serving prediction parity.

`development_errors_for_review.csv` is a local review list, not permission to
replace labels with predictions. No exact development input combinations had
conflicting labels; this does not establish label correctness. Individual
records, predictions, embeddings and error-review CSVs remain local.
