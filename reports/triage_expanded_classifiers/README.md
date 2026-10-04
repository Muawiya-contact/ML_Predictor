# Eleven additional classifier settings

All eleven predeclared CatBoost, CPU XGBoost and RBF SVM configurations completed
five fixed grouped folds. Verification recomputed every out-of-fold metric,
checked immutable data/embedding hashes and confirmed group separation.
No test records participate in this selection stage.

The best eligible new candidate is CatBoost depth 6, PCA-64: mean macro F1
87.23%, compared with 89.65% for the retained Logistic Regression. The conditional
paired group difference interval is negative; the challenger is not promoted.

See the [current joint report](../triage_four_level_round6/) for all six families,
retrospective comparisons and the 90% target assessment. `environment.json`
records actual CPU package versions; `protocol.json` was written before fitting.
