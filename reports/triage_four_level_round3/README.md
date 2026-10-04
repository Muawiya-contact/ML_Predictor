# Four-level SapBERT improvement comparison

The eight-page PDF retains the original 768-D/PCA-64 tables and matrices, adds
all 49 development configurations (245 fits), and reports tuned comparisons
for Logistic Regression, Random Forest and HistGradientBoosting. No additional
classifier families were used. All source labels, groups and splits are unchanged.

The constrained development selection is `lr_quad128_c100`.
Retrospective accuracy is 86.29%, macro precision
86.75%, macro recall 86.87% and
macro F1 86.80%. Emergency recall is
91.42%. These scores use the previously examined
1,999-row test set and supplied/recovered concepts; they are not a fresh
independent estimate or a measurement of live translation accuracy.

The family comparison selects the highest fused development F1 within each
classifier separately. It does not override the emergency-recall constraint
used to select the deployed model. Consult `family_selection.json` for settings.
The prior-round result is included so unchanged or worse metrics stay visible.

`verification.json` independently checks folds, group separation, source hashes,
selection, predictions and reported metrics. `serving_verification.json` checks
live encoder parity and all 1,999 predictions through the shared application
adapter. Bootstrap uncertainty is conditional on the selected predictions;
it does not account for repeated model selection. Fresh independent data is
needed to establish generalization. No labels were corrected or inferred.

See [reproduction instructions](../../experiments/triage_study/IMPROVEMENT.md).
The [initial comparison](../triage_four_level/) is retained for traceability.
Raw records, individual predictions, embeddings and error-review files remain
local and are not included here.
