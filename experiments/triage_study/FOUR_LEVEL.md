# Four-level SapBERT study

This is the current application study. Historical three-level scripts and
reports describe different targets and must not be combined with these scores.

## Protocol

- Workbook: `cardiac_multilingual_10000_4level_triage.xlsx`; 10,000 rows.
- Targets: 0 Emergency, 1 Urgent, 2 Standard, 3 Non-urgent.
- Inputs: Clinical_Concept and ten patient features (six measurements, Gender,
  Mode_of_Arrival, ECG_Status, AVPU). Complaint text also defines leakage groups.
- Triage_Label, Category and translation-processing metadata are excluded.
- The 4,290 missing concepts are recovered from the previous supplied CSV only
  after row-for-row agreement on all eleven complaint/patient input fields and
  every existing concept. Previous labels are never copied.
- Exact duplicate records are removed. Connected components of repeated
  normalized complaints OR concepts cannot cross holdout or CV boundaries.
- A fixed approximately 80/20 grouped stratified split uses seed 42. Five
  grouped development folds use seed 123. Every classifier shares the holdout.
- Frozen SapBERT revision `090663c3ae57bf35ffe4d0d468a2a88d03051a4d` uses CLS
  pooling, a 64-token limit and L2 normalization. No encoder fine-tuning occurs.
- PCA-64, numeric imputation/scaling and categorical encoding are fitted within
  each training fold; the final fit uses development rows only.
- The 12 fixed baselines are text/fused × 768/PCA-64 × LR/HGB/RF. Fused inputs
  have 790 or 86 columns. The same classifier settings apply across views.
- Ten fused PCA-64 configurations are compared across five development folds.
  Highest mean macro F1 wins; ties prefer higher emergency recall, then lower
  under-triage. Candidate ID is a deterministic final tie breaker. The selected
  configuration is saved before any holdout evaluation.
- All 12 baselines plus the selected configuration are evaluated once on the
  fixed holdout. No subsequent hyperparameter changes use its outcomes.
- Under-triage means predicted level > reference level (lower assigned urgency).
  Scores include accuracy, macro precision/recall/F1, emergency recall, weighted
  kappa, mean absolute error, under-/over-triage and four-class confusion matrices.
  Group-bootstrap intervals use 1,000 holdout-group resamples, seed 2026.

## Reproduce

Install the pinned research requirements and CPU PyTorch. Source files stay local;
reports contain aggregates, not individual records. Each preparation/training
command requires a fresh directory to prevent mixing stale and current results.

```bash
python experiments/triage_study/prepare_four_level_source.py \
  --source /path/to/cardiac_multilingual_10000_4level_triage.xlsx \
  --concept-reference /path/to/cardiac_multilingual_10000_filled.csv \
  --output output/results/four_level_source

python experiments/triage_study/four_level_study.py \
  --source output/results/four_level_source/prepared_source.csv \
  --output output/results/triage_four_level \
  --model-path /path/to/pinned/SapBERT/snapshot

python experiments/triage_study/verify_four_level_study.py output/results/triage_four_level
python experiments/triage_study/export_four_level_bundle.py \
  --source output/results/triage_four_level --output /tmp/four_level_bundle
python experiments/triage_study/build_four_level_pdf.py \
  --results output/results/triage_four_level \
  --output output/pdf/SapBERT_Four_Level_Comparison.pdf
```

Alternatively pass `--embedding-source /path/to/verified/cache` to the training
command. Reuse requires the exact concept-text order, checkpoint revision, array
shape and checksum. Frozen vectors contain no labels; every classifier and PCA
is still fitted anew. The completed study is self-contained and does not need
the old dataset or embedding cache to reproduce predictions or render the PDF.

The label-assignment response was “by using all”; the exact process and independent
per-record review are not documented. Evaluation measures agreement with workbook
labels on supplied/recovered concepts. It does not validate live Ollama translation
or clinical use. The literature table follows the requested layout and explicitly
separates the different binary KTAS task from this four-class cardiac task.

## Deliverables

`reports/triage_four_level/` contains the eight-page PDF, all exported figures and
aggregate tables/metadata. `triage_model_sapbert/` contains the selected fitted
classifier, structured transformer, PCA, manifest and matching saved metrics.
Raw records, row-level predictions, embeddings and scratch files remain local
under `output/results/`; they are excluded from the PR.
