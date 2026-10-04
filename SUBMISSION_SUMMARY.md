# Article methodology: current four-level study

The selected model uses frozen SapBERT with paired complaint input, PCA-64,
50 patient features and 19 explicit original-complaint details. Balanced
Logistic Regression C=100 receives 133 inputs and predicts levels 0, 1, 2 and 3.

1. Retain all 10,000 supplied records and their four labels. Preserve the original
   8,001 development / 1,999 test split and five grouped development folds.
   The provider has no label-assignment rules available; no labels are replaced.
2. Recover missing clinical concepts only after exact matching of the original
   complaint/patient fields. Exclude targets and processing metadata from inputs.
3. Encode the clinical concept + [SEP] + original complaint with the pinned
   frozen SapBERT checkpoint, CLS pooling, L2 normalization and a 128-token limit.
4. Fit imputation, categorical encoding, scaling and PCA within each training
   fold. Add quadratic numeric patient terms and 19 explicit complaint details.
5. Compare the original classifiers, CatBoost, XGBoost, SVM, cumulative ordinal
   models, neural heads, cubic features, paired text and fixed probability blends.
   The 133 configurations contain 109 trainable settings and 24 blends: 545
   training fits plus 120 reused-probability evaluations across five folds.
6. Select by development macro F1 subject to the original emergency-recall
   threshold. Freeze the choice before the final retrospective evaluation.
7. Report accuracy, macro precision/recall/F1, individual class scores, confusion
   matrices, under-/over-triage, weighted kappa, MCC and probability diagnostics.
   The selected model reaches 90.60% accuracy and 90.99% macro F1 on previously
   examined test rows. Not every class score exceeds 90%; emergency recall is
   94.36%, versus 95.10% previously. These are not independent clinical results.
8. Export the fitted transformations and model. Verify all 1,999 probabilities
   and predictions through the same application adapter before promotion.
9. Describe the six-tab GUI separately: local Ollama translation, anatomical
   checking, paired encoder input and the selected classifier. Saved concepts
   are used in the study; live translation accuracy is not measured here.

Use the [current report](reports/triage_four_level_round7/) and active model
manifest. SapBERT itself was not fine-tuned. Literature tables use other tasks
and provide context, not evidence of superiority over those published systems.
