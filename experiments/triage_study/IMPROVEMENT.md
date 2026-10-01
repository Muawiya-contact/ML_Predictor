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

After joint selection, report the highest-CV-F1 fused configuration within each
of the three families, then independently check all reported metrics:

```bash
.venv/bin/python experiments/triage_study/compare_tuned_families.py \
  --source output/results/triage_four_level_round3 \
  --original output/results/triage_four_level
.venv/bin/python experiments/triage_study/verify_improvement.py \
  --source output/results/triage_four_level_round3 \
  --original output/results/triage_four_level
```

Family comparisons are descriptive and do not override constrained deployment
selection. Use the verified round-three source with `export_improved_bundle.py`,
`verify_improved_serving.py` and `build_improvement_pdf.py` to produce the current
bundle and report. Retain an archived initial bundle as the export's `--incumbent`.
The PDF builder's `--original` argument is `reports/triage_four_level` because it
also needs the verified literature-source metadata stored there.


## Original complaint details and learning curves

The development-only audit checks missing/different duration mentions and other
lexical signals without changing labels. It retains the three classifier
families and fixed SapBERT embeddings. Nineteen explicit bilingual indicators
are scaled inside each training fold, then appended to the existing features.
They represent mentions, not adjudicated clinical findings. Original complaint
text must be retained separately from the English text used by SapBERT.

```bash
python experiments/triage_study/learning_and_detail_audit.py \
  --source output/results/triage_four_level \
  --incumbent output/results/triage_four_level_round3 \
  --output output/results/triage_learning_detail_audit
python experiments/triage_study/verify_learning_audit.py \
  --source output/results/triage_learning_detail_audit \
  --original output/results/triage_four_level \
  --incumbent output/results/triage_four_level_round3
python experiments/triage_study/finalize_detail_comparison.py \
  --source output/results/triage_learning_detail_audit \
  --original output/results/triage_four_level \
  --incumbent output/results/triage_four_level_round3 \
  --output output/results/triage_four_level_round4
python experiments/triage_study/verify_improvement.py \
  --source output/results/triage_four_level_round4 \
  --original output/results/triage_four_level
```

The audit runs 90 fits: 60 nested whole-group learning-curve fits and 30
full-size detail-feature fits. The original full-size baselines are reproduced
and checked. Combined with earlier rounds, six new settings make 55 settings
and 275 full-size fold fits. The original emergency-recall threshold remains
unchanged. Selection precedes all retrospective test evaluation.

Use round four with the existing export, serving-verification and full-report
commands above. The export includes a fourth fitted artifact, detail_scaler.pkl,
and verifies saved probabilities as well as predicted classes. GUI/CLI wrappers
retain Raw_Complaint; direct prediction callers must supply it for this bundle.
The clinical labels, encoder and original split remain unchanged.

```bash
python experiments/triage_study/build_learning_pdf.py \
  --source output/results/triage_learning_detail_audit \
  --comparison output/results/triage_four_level_round4 \
  --output output/pdf/SapBERT_Data_Quality_and_Learning_Curves.pdf
```

The review queue, group membership lists and all individual predictions remain
local. Learning curves describe the observed sample sizes; they do not promise
an accuracy at 20,000 records. Fresh independently reviewed data is required to
confirm generalization after repeated development searches.
