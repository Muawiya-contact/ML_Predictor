# Previous versus latest SapBERT comparison

[Read the comparison PDF](SapBERT_Previous_vs_Latest_Comparison.pdf). This is an
explanatory companion to the [current model report](../triage_four_level_round5/),
not a replacement model or a new training run.

The 19-page comparison includes all 12 old and current fixed conditions, 24
baseline matrices with counts and row percentages, latest tuned-family results,
class scores, the complete 68-setting development table, audit context and a
plain-language assessment of each change. All pages were rendered and inspected.

Both reports use 1,999 test records, but only 438 records are shared between the
test sets. The label transition is not a one-to-one renumbering. Lower raw counts
alone do not mean worse performance; percentage scores concern different targets.
The latest four-level refinement is a modest observed gain with uncertainty.

`verification.json` identifies the two source PDFs and records recomputation of
all 24 baseline matrices and headline metrics. `score_differences.csv` contains
unrounded numerical deltas. Individual records are not included.

To reproduce after generating the saved source experiments and current report:

```bash
python experiments/triage_study/build_historical_comparison_pdf.py \
  --previous-pdf /path/to/SapBERT_Full768_PCA64_Comparison-1.pdf
```

Required local evidence directories: `output/results/triage_fixed_full_pca`,
`output/results/triage_10000_research`, `output/results/triage_four_level`,
`output/results/triage_four_level_round5`, and the learning/detail audit outputs.
The script reads saved evidence; it does not retrain or change any labels.
