# Current four-level SapBERT comparison

Use [SapBERT_Final_Report.pdf](SapBERT_Final_Report.pdf) as the current report.
It retains the original full-768/PCA-64 comparisons, includes all six classifier
families, all 79 development settings, confusion matrices, class scores, the
learning/error audits and a new embedding-geometry diagnostic.

The expansion adds eleven CatBoost, CPU XGBoost and RBF SVM settings: 55 new
five-fold fits, bringing the combined search to 79 settings and 395 fits. The
90% target was **not achieved** across accuracy and macro precision/recall/F1.
The previous Logistic Regression remains the best eligible development model:
SapBERT PCA-128, 50 patient features and 19 original-complaint features, balanced
C=10. The deployed bundle is retained without replacing its fitted artifacts.

| Classifier | Mean development macro F1 | Retrospective accuracy | Retrospective macro F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 89.65% | 89.29% | 89.73% |
| HistGradientBoosting | 88.07% | 88.19% | 88.68% |
| Random Forest | 81.96% | 82.29% | 82.58% |
| CatBoost | 87.23% | 87.09% | 87.49% |
| XGBoost | 86.39% | 86.09% | 86.60% |
| RBF SVM | 85.49% | 85.64% | 86.17% |

Each family representative is selected by fused development macro F1, not by
these test scores. The deployment additionally requires the original emergency
recall threshold (92.3817%). The same 1,999 test records were examined previously;
their scores are retrospective, not fresh independent confirmation.

The retained model's macro precision is 89.67%, macro recall 89.81% and emergency
recall 95.10%. No source labels, records, partitions or frozen SapBERT vectors
were changed. Live Ollama translation accuracy is not measured by these scores.

The best new eligible challenger, CatBoost, has a pooled development F1 difference
of -2.42 percentage points versus the incumbent (conditional paired group 95%
interval -3.16 to -1.70 points). That interval excludes repeated-selection
uncertainty. Additional classifier families did not improve the current ranking.

The [embedding diagnostic](../triage_embedding_diagnostics/) samples 1,200
development groups. Similarities and silhouettes show weak separation by supplied
triage level in text embeddings alone. This is descriptive geometry, not a
performance ceiling; patient measurements and detail features also matter.

Validation: all 55 new OOF fits and 395 combined CV records verified, source
hashes/group boundaries checked, all family metrics/matrices recomputed, and
all 1,999 active-model predictions/probabilities reproduced through the shared
adapter. Thirteen live embeddings and a live English prediction match. The
45 targeted unit tests and real six-tab GUI audit passed. The PDF's 15 pages
were rendered and visually inspected before publication.

[Reproduce the comparison](../../experiments/triage_study/EXPANDED_CLASSIFIERS.md).
[New-candidate protocol and results](../triage_expanded_classifiers/).
`deployment_decision.json` records the retained artifact hashes and explicit
target assessment. Individual records, error queues and OOF arrays remain local.

The [historical comparison](../triage_historical_comparison/) retains the older
three-level versus round-five explanation. Its selected four-level model and
headline scores are unchanged; it does not include these three new families.
