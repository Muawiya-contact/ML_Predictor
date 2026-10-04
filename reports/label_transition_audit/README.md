# Why the previous 99% scores do not match the four-level study

The uploaded seven-page `SapBERT_Full768_PCA64_Comparison-1.pdf` describes three target levels. All twelve original accuracy, macro precision, macro recall and macro F1 results and confusion matrices were recomputed from saved predictions and matched. The literature rows describe a different binary task and are not a directly comparable target.

Across the old and current prepared datasets, all 10,000 rows have identical patient inputs, complaints and clinical concepts, and byte-identical SapBERT embeddings. Group identities are unchanged. Targets are different: old class 1 splits into 2,040 new Emergency and 1,349 new Urgent cases. This is not a one-to-one renumbering. The grouped stratified split was recreated for the new labels: 3,122 records change development/test membership even though both studies contain 8,001 development and 1,999 test rows.

## Controlled label-task comparison

Both label tasks below use the same current split, PCA-64, 22 patient features, feature preprocessing and original fixed classifier settings. Only the target labels change. These six retrospective diagnostic fits use previously examined test rows; they are not fresh independent validation and do not select or replace the deployed model. The three new-label runs reproduce every original four-level baseline prediction.

| Classifier | Label task | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | old three level | 98.30% | 97.95% | 98.27% | 98.11% |
| Logistic Regression | current four level | 82.44% | 83.01% | 83.22% | 83.03% |
| HistGradientBoosting | old three level | 99.20% | 99.31% | 98.87% | 99.09% |
| HistGradientBoosting | current four level | 82.09% | 83.28% | 82.54% | 82.76% |
| Random Forest | old three level | 99.10% | 98.63% | 99.32% | 98.96% |
| Random Forest | current four level | 77.89% | 78.47% | 78.65% | 78.35% |

The label-task change produces a large gap even after removing the split difference. This supports a change in task predictability, not a broken encoder. It does not prove that the new labels are wrong or establish an accuracy ceiling. The deployed tuned model remains 86.29% accuracy and 86.80% macro F1; the fixed-settings controls above are intentionally different from that tuned model.

## What to address next

- Keep the three-level and four-level results separate in the article. The earlier 99% is not a valid expected score for a different target task.
- Review four-level definitions and ambiguous development examples with an appropriate domain reviewer. Label-assignment rules are unavailable; do not replace labels with predictions to increase scores.
- Investigate whether the new decisions require information absent from the current structured inputs or compressed clinical concepts. Any added feature needs a reproducible inference-time source.
- After justified data/feature corrections, repeat grouped development validation with the same three classifiers and confirm with fresh independently labelled test data. Repeated tuning on known test results cannot provide that confirmation.

## Reproduction

```bash
.venv/bin/python experiments/triage_study/audit_label_transition.py \
  --old-source output/results/triage_10000_research \
  --current-source output/results/triage_four_level \
  --output output/results/label_transition_audit_fresh
```

`all_twelve_score_comparisons.json` preserves all original and four-level baseline scores. `protocol.json` records identities, class counts and transitions; `results.json` contains the six controlled results. Raw source data and individual records are not published.
