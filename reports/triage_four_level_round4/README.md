# Four-level SapBERT: original complaint details

Current model: balanced Logistic Regression C=100, frozen SapBERT PCA-128,
50 patient features and 19 explicit original-complaint details (197 inputs).
Labels remain 0 Emergency, 1 Urgent, 2 Standard and 3 Non-urgent.

Five-fold development macro F1 is 89.38%, versus the prior 86.57%. The combined
search contains 55 settings and 275 full-size fold fits, retaining the original
emergency-recall constraint. Selection is frozen before retrospective scoring.

On the previously examined 1,999 records: accuracy 88.99%, macro precision
89.37%, macro recall 89.50%, macro F1 89.43%, emergency recall 93.87% and
under-triage 4.65%. These are retrospective results, not independent clinical
validation. Labels and encoder were not changed. Saved English concepts feed
SapBERT; original complaints provide detail features. Live translation accuracy
was not evaluated.

The eight-page PDF retains all twelve baseline full-768/PCA-64 comparisons,
tuned-family scores, confusion matrices, class reports and literature context.
All 275 fits and reported scores were independently checked. The exported
adapter reproduces all 1,999 saved predictions and probabilities. Thirteen
live embedding checks and one live English prediction passed. All 37 unit
tests and the real four-level GUI audit passed. Every PDF page was rendered
and visually inspected.

See ../triage_learning_detail_audit/ for the separate development-only audit,
learning curves and feature comparison. Individual records remain local.
Reproduction commands are in ../../experiments/triage_study/IMPROVEMENT.md.
