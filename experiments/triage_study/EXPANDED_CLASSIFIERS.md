# Additional classifier comparison

This follow-up was authorized after the three-family study. It tests CatBoost,
CPU XGBoost and RBF SVM alongside the existing Logistic Regression, Random Forest
and HistGradientBoosting results. The 10,000 records, four labels, SapBERT cache,
8,001/1,999 partition, five grouped folds and emergency-recall threshold stay fixed.
No new encoder or translator is substituted in this experiment.

Eleven additional settings are declared in `expanded_classifiers.py` before
fitting: CatBoost depth 4/6, XGBoost depth 4/6, each with PCA-64/128, and RBF SVM
C=1/10/100 with PCA-128 and quadratic patient terms. All use the same 19
original-complaint details. Imputation, scaling and PCA fit on training-fold
rows only; balanced sample weights are computed from those rows only.

The target is at least 90% accuracy, macro precision, macro recall and macro F1.
It is a target, not an acceptance rule that changes labels or test membership.
Every fit, including unsuccessful settings, is retained. Selection uses mean
five-fold macro F1 subject to the original emergency-recall constraint; test
scores never choose the model. A losing challenger does not replace deployment.

Install the optional experiment dependencies into the existing pinned study
environment (these are not required by the retained Logistic Regression GUI):

```bash
python -m pip install -r experiments/triage_study/requirements-expanded.txt
python experiments/triage_study/expanded_classifiers.py \
  --original output/results/triage_four_level \
  --output output/results/triage_expanded_classifiers
python experiments/triage_study/verify_detail_refinement.py \
  --source output/results/triage_expanded_classifiers \
  --original output/results/triage_four_level \
  --incumbent output/results/triage_four_level_round5
python experiments/triage_study/finalize_improvement.py \
  --main output/results/triage_four_level_round5 \
  --extra output/results/triage_expanded_classifiers \
  --original output/results/triage_four_level \
  --output output/results/triage_four_level_round6
python experiments/triage_study/compare_tuned_families.py \
  --source output/results/triage_four_level_round6 \
  --original output/results/triage_four_level
python experiments/triage_study/verify_improvement.py \
  --source output/results/triage_four_level_round6 \
  --original output/results/triage_four_level
python experiments/triage_study/embedding_diagnostics.py \
  --original output/results/triage_four_level \
  --bundle triage_model_sapbert \
  --output output/results/triage_embedding_diagnostics
python experiments/triage_study/build_improvement_pdf.py \
  --source output/results/triage_four_level_round6 \
  --original reports/triage_four_level \
  --audit output/results/triage_learning_detail_audit \
  --investigation output/results/triage_error_investigation \
  --expanded output/results/triage_expanded_classifiers \
  --geometry output/results/triage_embedding_diagnostics \
  --output output/pdf/SapBERT_Final_Report.pdf
```

Use fresh experiment destinations. The joint comparison contains 79 settings
and 395 five-fold fits, reusing 340 already completed fits rather than rerunning
them. Family representatives are descriptive highest-F1 fused settings; they do
not override the constrained deployment selection. The existing test records
were examined previously, so test comparisons remain retrospective.

All classification paths use `classes_[predict_proba(X).argmax(axis=1)]`.
This matters for SVM, whose `predict()` decision can differ from its calibrated
probability winner. The exporter checks every saved probability and prediction.
Optional CatBoost/XGBoost bundles record and enforce their package version
before unpickling. SVM probability support is tied to pinned scikit-learn 1.9;
a later upgrade requires retraining and parity checks.

The report records whether the aggregate 90% target was met. It does not imply
that the highest tested score is a theoretical performance ceiling. Source
records, individual errors and OOF arrays remain local; publish aggregate
results, plots, the protocol and verification only.

The embedding diagnostic samples 1,200 development groups with a fixed seed.
It reports within/between triage-label cosine similarity and silhouette in
768-D, PCA-64 and PCA-128 spaces, plus a two-component plot. This describes
geometry, not cross-validated accuracy; the PCA was fitted on all development
rows. Centring changes cosine baselines, and low separation does not establish
a prediction ceiling or incorrect labels. It does not select a classifier.
