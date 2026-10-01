# Four-level SapBERT comparison

The eight-page PDF includes both six-row baseline tables, full-768/PCA-64
accuracy and precision plots, twelve four-class confusion matrices, the
literature comparison, five-fold development CV and the selected model's
held-out class scores and matrix. All 29 figures/tables are exported as PNGs.

The selected fused PCA-64 Logistic Regression (C=10, balanced training weights)
scored 85.19% accuracy, 85.71% macro precision, 85.90% macro recall and 85.80%
macro F1 on 1,999 held-out records. Emergency recall was 92.16%; under-triage
was 7.60%. Selection used development CV only. The matching application bundle
is `triage_model_sapbert/` with labels 0, 1, 2 and 3.

See [the complete protocol](../../experiments/triage_study/FOUR_LEVEL.md).
Raw source records, individual predictions and embeddings remain local.
The source workbook has 10,000 records, with 4,290 exactly matched concept
recoveries. Label provenance and the distinction between supplied concepts and
live translations are described in the PDF; these are not clinical validation.

`metrics.csv` contains the twelve fixed conditions; `selected_results.json`
contains the development-selected model. `cross_validation.csv` contains all
50 development fits. Verification, source identity, software versions and GUI/PDF
checks are included alongside them. Literature values use another task and are
context only, not a direct superiority benchmark.
