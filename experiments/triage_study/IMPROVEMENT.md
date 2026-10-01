# Four-level improvement round

This round preserves all source labels, SapBERT settings and original group
partitions. It searches 29 initial configurations and six separately predeclared quadratic
interaction configurations over the same five development folds (175 fits total). A maximum of three separate fold processes run concurrently;
preprocessing is fitted independently inside each fold.

Candidates include Logistic Regression with C=10/100/1000, balanced/unbalanced
training and PCA-64/128/256; whitened PCA with C=0.1; regularized HGB with 7/15
leaves; and Random Forest. Structured-only models are controls. SapBERT remains
frozen. Quadratic numeric terms are fitted inside each fold; no new labels or clinical
facts are invented. The provider has no label-assignment rules available.

Selection uses highest mean development macro F1 subject to emergency recall
being within one percentage point of the incumbent. Accuracy breaks ties. This
is an experimental selection criterion, not a clinical safety guarantee. The
incumbent is reproduced on the same folds. All candidates are reported.

The supplemental search is defined before inspecting the new retrospective
results. The joint choice uses only development CV. The old test set was already examined. New results on it are **retrospective**,
not an untouched validation claim. Selection is frozen before these scores are
computed. Fresh, independently labelled records are needed for confirmation.
The OOF paired group bootstrap is conditional on the selected predictions; it
does not account for selection optimism from the expanded search.

Run from the repository root with the pinned project environment:

```bash
.venv/bin/python experiments/triage_study/improve_four_level.py \
  --source output/results/triage_four_level \
  --output output/results/triage_four_level_improved_parallel
.venv/bin/python experiments/triage_study/improvement_interactions.py \
  --original output/results/triage_four_level \
  --output output/results/triage_four_level_interactions
.venv/bin/python experiments/triage_study/finalize_improvement.py \
  --main output/results/triage_four_level_improved_parallel \
  --extra output/results/triage_four_level_interactions \
  --original output/results/triage_four_level \
  --output output/results/triage_four_level_round2
.venv/bin/python experiments/triage_study/verify_improvement.py \
  --source output/results/triage_four_level_round2 \
  --original output/results/triage_four_level
.venv/bin/python experiments/triage_study/export_improved_bundle.py \
  --source output/results/triage_four_level_round2 \
  --original output/results/triage_four_level \
  --incumbent triage_model_sapbert \
  --output output/results/triage_four_level_round2/exported_bundle
.venv/bin/python experiments/triage_study/build_improvement_pdf.py \
  --source output/results/triage_four_level_round2 \
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

## Refinement within the same three classifier families

`improvement_refinement.py` predeclares 14 additional configurations (70 fits):
balanced LR C=30/300 at PCA-64/128; quadratic LR C=100/300 at PCA-128/256;
HGB PCA-64 with 7/31 leaves and 800/400 iterations; RF PCA-64/128 with 500
trees, square-root feature subsampling and minimum leaf size 1/3. No new
classifier family, encoder, source labels or partition is introduced.

```bash
.venv/bin/python experiments/triage_study/improvement_refinement.py \
  --original output/results/triage_four_level \
  --output output/results/triage_four_level_refinement
.venv/bin/python experiments/triage_study/finalize_improvement.py \
  --main output/results/triage_four_level_round2 \
  --extra output/results/triage_four_level_refinement \
  --original output/results/triage_four_level \
  --output output/results/triage_four_level_round3
.venv/bin/python experiments/triage_study/verify_improvement.py \
  --source output/results/triage_four_level_round3 \
  --original output/results/triage_four_level
```

The joint ranking keeps the **original** emergency-recall threshold; it is not
relaxed with each round. The previous winner remains eligible. All 49 settings
are retained in the comparison even when refinement fails to improve scores.
The previously examined test set remains retrospective. Repeated development
search is not a substitute for fresh independent validation.
