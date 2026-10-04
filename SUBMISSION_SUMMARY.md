# Article methodology: current four-level study

The study uses the supplied 10,000-row cardiac workbook and predicts four levels:
0 Emergency, 1 Urgent, 2 Standard and 3 Non-urgent. SapBERT stays frozen; the
classifiers and preprocessing are retrained. The current application uses
PCA-128 plus 50 patient features and 19 original-complaint detail features with
balanced Logistic Regression (C=10).

1. Validate the workbook's numeric levels against its class names. Keep all
   supplied labels unchanged. The provider has no assignment rules available.
2. Recover 4,290 missing clinical concepts only after matching all eleven
   complaint/patient fields and populated concepts against the earlier source.
   Exclude target names and processing metadata from model inputs.
3. Retain the original 8,001 development / 1,999 test split. Connected groups of
   repeated complaints or concepts cannot cross split or validation boundaries.
4. Encode concepts with the pinned SapBERT checkpoint: CLS pooling, 64-token
   truncation and L2 normalization produce 768-dimensional embeddings.
5. Fit imputation, numeric scaling, categorical encoding and PCA within each
   training fold. The selected model combines 128 text components with 50
   patient features (35 numeric linear/quadratic terms and 15 categorical indicators),
   plus 19 explicit bilingual complaint-detail features, giving 197 inputs.
6. Retain the original 12 comparisons across three classifiers, full/PCA-64
   embeddings and text-only/fused inputs. Expand development selection to 79
   configurations, including PCA-128/256, regularization, class balancing,
   whitening, structured-only controls, quadratic patient interactions and
   original/concept detail-feature comparisons. The final expansion adds
   CatBoost, CPU XGBoost and RBF SVM; none displaces the selected Logistic Regression.
7. Evaluate every candidate across five grouped folds. Select by mean macro F1
   while requiring emergency recall within one percentage point of the original baseline;
   accuracy breaks ties. This is an experimental criterion, not a clinical guarantee.
8. Freeze selection before the final retrospective comparison. Report accuracy,
   macro precision/recall/F1, class scores, confusion matrices, under-/over-triage,
   weighted kappa, mean level error, MCC and probability diagnostics. The previously examined test set is not
   fresh independent validation. Conditional paired development bootstrap intervals exclude
   model-selection uncertainty. New data is needed to confirm generalization.
9. Export the fitted transformations and classifier without further fitting on
   test records. Verify all 1,999 predictions through the shared application
   adapter and check live encoder parity before promoting the bundle.
10. Describe the six-tab GUI separately: local Ollama translation, deterministic
    anatomical checking and the selected classifier. Saved experimental metrics
    use supplied/recovered concepts and do not measure live translation accuracy.

Use [the current report](reports/triage_four_level_round6/) and
`triage_model_sapbert/triage_metrics.json`. Earlier three-level, MiniLM and initial
four-level results remain distinct experiments. No clinician review or clinical
validation is inferred. Literature values use another binary KTAS task and are
context rather than a direct superiority benchmark.

The expanded comparison did not reach 90% across all four aggregate metrics.
The retained model has 89.29% retrospective accuracy and 89.73% macro F1.
Additional classifiers alone did not improve on its development ranking.
