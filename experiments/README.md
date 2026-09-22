# Article Sections 3.4–3.6

This is a separate research experiment. It does not replace the deployed model.
The user requested Sentence-BERT with 768-dimensional outputs; the selected
checkpoint is `sentence-transformers/all-mpnet-base-v2` (English, mean pooling,
L2-normalized output). The deployed multilingual MiniLM checkpoint is 384-D.
Model documentation: https://huggingface.co/sentence-transformers/all-mpnet-base-v2

## Reproduce Sections 3.4 and 3.5

Run from the repository root with its virtual environment:

```bash
.venv/bin/python -m pip install -r experiments/requirements.txt
.venv/bin/python experiments/prepare_sbert.py
.venv/bin/python experiments/compare_classifiers.py \
  --data output/results/sbert_inputs/labelled_rows.csv \
  --embeddings output/results/sbert_inputs/embeddings.npz \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --output output/results/sbert_text_only
```

`prepare_sbert.py` records the resolved checkpoint commit. Pass that commit with
`--revision` to reproduce the embedding run. It downloads weights only; all text
encoding is on CPU. Existing English translations are reused without learned
stop-word filtering or fine-tuning. Each NPZ vector is explicitly keyed to a CSV
row ID. The original 50-by-768 array has no established triage-label mapping and
cannot support 64-component training PCA; it is not used for classification.

To run both variants and generate combined Section 3.4/3.5 tables after preparing the embeddings:

```bash
.venv/bin/python experiments/run_comparisons.py
```

For a fusion ablation, repeat the comparison with `--structured` and a different
output directory, such as `output/results/sbert_fusion`. This appends train-fitted
numeric/categorical patient features to each text representation. The same seed
and rows guarantee identical test partitions for both runs. Text-only inputs have
768 or 64 columns; fusion inputs additionally include the structured block.

The three estimators are Logistic Regression, Histogram Gradient Boosting and
Random Forest. Logistic Regression and Random Forest use balanced class weights; Histogram
Gradient Boosting uses its default unweighted loss, as in `src/models.py`.
Weighted binning was prohibitively slow in the installed scikit-learn build. The final CPU budget is 100 boosting iterations
with at most 15 leaves per tree, 200 Random Forest trees, and up to 3,000 Logistic
Regression iterations. The same settings are used for every representation. No hyperparameter search or model
selection uses the test set. The protocol is a single stratified 80/20 split with
seed 42. PCA centering and components, imputation, numeric scaling, and one-hot
vocabulary are fitted using training rows only. Embeddings use frozen weights.
Rates and scores in CSV tables are fractions between 0 and 1, not percentages.
Zero-denominator precision is reported as zero. Confusion matrices contain counts
with true classes in rows and predicted classes in columns. Smaller triage labels
are more urgent; predicting a larger label counts as under-triage.

Each run produces CSV, Markdown and LaTeX summary tables, per-class metrics,
accuracy and macro-precision graphs, six confusion matrices (PNG and CSV), row-level
predictions, split membership, PCA parameters and a manifest with input hashes,
versions and classifier settings. Output directories must be empty to prevent
accidental overwriting of earlier runs. Timings measure fitting and prediction,
excluding embedding generation and PCA; they are not end-to-end app latency.

The data is synthetic cardiac data with previously generated English translations.
Results evaluate classifier performance on this dataset, not live Roman Urdu
translation or clinical effectiveness. Random row splitting may share repeated
complaint phrases between partitions. A held-out clinical dataset and grouping by
patient/template would be separate evaluations; no clinical validation is claimed.

## Section 3.6: Comparing with State-of-the-Art

The local article has only this heading, and no notebook defines additional
requirements. Published comparison methods, sources and evaluation protocols need
to be specified before this section can be completed. Do not treat the three local
classifiers as published state-of-the-art evidence or compare unrelated dataset
accuracy values as if they were measured on the same test set. A source table is
provided in `output/results/section_3_6_comparison_template.csv` for the required
method, citation, dataset, target, split and reported scores. It deliberately
contains no invented literature results.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -p test_research_comparison.py -v
```

## PDF report

The six-page report is `output/pdf/SBERT_Classifier_Comparison.pdf`. Rebuild it
from the saved results with `python experiments/export_results_pdf.py`.
