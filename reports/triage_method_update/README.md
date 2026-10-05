# Semantic concepts and method improvements

[Combined report](SapBERT_Method_Improvement_and_Semantic_Concepts.pdf): preserves the existing eight-page selected-model report and appends eight pages with symptom-mention scatter plots, exact baseline comparisons, method changes and a manuscript-ready explanation.

Use Figure 7 on page 9 for the semantic analysis. The retained urgency-coloured plot on page 5 is an auxiliary analysis, not the original manuscript's semantic-concept experiment. Four fixed samples of 100 development complaints are shown; all seeds are retained. The new scatter plots use paired-text SapBERT and a label-blind diagnostic PCA, not the deployed PCA-64. Labels are literal symptom mentions, not verified diagnoses.

The 83.64% reference is full-768 HistGradientBoosting. The earlier PCA-64 Logistic Regression reference is 82.44%; current Logistic Regression is 90.60%. Both reference prediction files were checked against the same 1,999 record IDs and supplied labels. Differences compare complete configurations and do not isolate each change's causal contribution. Test results are retrospective.

Figures are available individually in PNG and vector PDF, with captions, aggregate metrics and provenance. No individual records are included. No trained model or GUI was changed.

Reproduce with saved local study caches:

```bash
python experiments/triage_study/build_method_update_report.py
```

Validation: all 16 pages rendered and visually inspected; first eight pages preserved; source/configuration hashes checked; four 100-record samples verified; comparison IDs/labels identical.

## Semantic-statistics update

Pages 13-16 add the 20-resample distance/effect-size/ANOSIM/silhouette table, dependency-aware statistical interpretation, verified literature context, and current LR/HGB/RF metrics and matrices. Silhouette values remain in [-1, 1]; negative results are retained. There are 80 run/space tests, each with 4,999 record-label permutations and Holm correction across all 80. Mention-derived groups are exploratory, not independently verified clinical categories. Overlapping resamples are not treated as independent datasets for t-test/Wilcoxon inference.

Reproduce after generating the 12-page method report:

```bash
python experiments/triage_study/semantic_statistics.py
python experiments/triage_study/build_statistics_report.py
```

The same output PDF and `output/pdf/img.zip` are refreshed. `statistics/` contains aggregate measurements, protocol and verification only. Classifier and encoder artifacts are unchanged.
