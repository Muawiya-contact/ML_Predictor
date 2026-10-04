# Semantic concepts and method improvements

[Combined report](SapBERT_Method_Improvement_and_Semantic_Concepts.pdf): preserves the existing eight-page selected-model report and appends four pages with symptom-mention scatter plots, exact baseline comparisons, method changes and a manuscript-ready explanation.

Use Figure 7 on page 9 for the semantic analysis. The retained urgency-coloured plot on page 5 is an auxiliary analysis, not the original manuscript's semantic-concept experiment. Four fixed samples of 100 development complaints are shown; all seeds are retained. The new scatter plots use paired-text SapBERT and a label-blind diagnostic PCA, not the deployed PCA-64. Labels are literal symptom mentions, not verified diagnoses.

The 83.64% reference is full-768 HistGradientBoosting. The earlier PCA-64 Logistic Regression reference is 82.44%; current Logistic Regression is 90.60%. Both reference prediction files were checked against the same 1,999 record IDs and supplied labels. Differences compare complete configurations and do not isolate each change's causal contribution. Test results are retrospective.

Figures are available individually in PNG and vector PDF, with captions, aggregate metrics and provenance. No individual records are included. No trained model or GUI was changed.

Reproduce with saved local study caches:

```bash
python experiments/triage_study/build_method_update_report.py
```

Validation: all 12 pages rendered and visually inspected; first eight pages preserved; source/configuration hashes checked; four 100-record samples verified; comparison IDs/labels identical.
