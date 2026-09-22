# Sections 3.4 and 3.5: classifier results

Encoder: `sentence-transformers/all-mpnet-base-v2` at commit `e8c3b32edf5434bc2275fc9bab85f82640a19130`.
Frozen, normalized 768-dimensional embeddings of existing English translations.
Data: 2,252 synthetic cardiac records; 1,801 training and 451 test rows; seed 42.
All runs use identical test rows. PCA and structured preprocessing use training rows only.

| Features | Representation | Classifier | Accuracy | Macro precision | Macro F1 |
|---|---|---|---:|---:|---:|
| text_only | full_768 | LogisticRegression | 58.09% | 53.49% | 54.85% |
| text_only | full_768 | HistGradientBoosting | 57.43% | 53.29% | 47.98% |
| text_only | full_768 | RandomForest | 55.65% | 55.53% | 51.08% |
| text_only | pca_64 | LogisticRegression | 55.65% | 50.68% | 51.78% |
| text_only | pca_64 | HistGradientBoosting | 56.98% | 51.39% | 47.34% |
| text_only | pca_64 | RandomForest | 54.32% | 47.86% | 46.53% |
| fusion | full_768 | LogisticRegression | 82.48% | 77.03% | 78.68% |
| fusion | full_768 | HistGradientBoosting | 82.93% | 82.08% | 76.97% |
| fusion | full_768 | RandomForest | 72.73% | 70.12% | 64.37% |
| fusion | pca_64 | LogisticRegression | 82.26% | 76.58% | 78.28% |
| fusion | pca_64 | HistGradientBoosting | 80.71% | 81.54% | 77.27% |
| fusion | pca_64 | RandomForest | 80.71% | 79.13% | 73.06% |

## Files

- `section_3_4_metrics.csv` / `section_3_4_table.tex`: original 768-D comparison.
- `section_3_5_metrics.csv` / `section_3_5_table.tex`: PCA-64 comparison.
- `all_comparison_metrics.csv`: all twelve runs, including fusion ablation and timings.
- `sbert_text_only/` and `sbert_fusion/`: accuracy/precision graphs, confusion matrices, per-class scores and predictions.
- `sbert_inputs/embedding_manifest.json`: exact checkpoint and input provenance.

Fusion appends patient features, so its total classifier input is larger than 768 or 64.
CSV scores are fractions; the table above displays percentages. These are single-split results, not cross-validation means.
The dataset is synthetic, and repeated complaint phrases may cross the random row split.
The results do not establish clinical safety or measure the live translation pipeline.

## Section 3.6 — pending source requirements

The article contains only “Comparing with State-of-the-Art”; no notebook or published comparator list was supplied.
`section_3_6_comparison_template.csv` records the required comparison fields. No literature scores have been invented.

Reproduction instructions: [experiments/README.md](../../experiments/README.md).
