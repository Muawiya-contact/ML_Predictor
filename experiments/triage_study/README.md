# Frozen-encoder triage classifier study

This isolated experiment evaluates the uploaded 10,000-record CSV with AI-assigned labels 1–3. It does not change the deployed app, translator, confidence cap, safety checks, or saved application models. The encoders are **frozen**: this is classifier retraining and hyperparameter selection, not language-model fine-tuning.

## What changed

- Retain 12 input fields plus `Labels`; exclude constant Category and processing metadata. An invalid arrival value becomes missing. No labels are rewritten.
- Connect records sharing a normalized complaint **or** concept, then use grouped splits: 8,001 development and 1,999 held-out records for this input. Exact text overlap is prevented; semantically related templates may remain.
- Fit imputation, scaling, categorical encoding and PCA separately within each training fold.
- Compare Logistic Regression, Hist Gradient Boosting and Random Forest on structured, text and combined features. Compare frozen SapBERT/MPNet 768-D and multilingual MiniLM 384-D, PCA 32/64/128/full, class balancing, regularization, tree settings, derived vital-sign features, text weights and probability combinations.
- Screen 136 configurations with three-fold grouped CV; refine 22 with five-fold grouped CV. Freeze selection before final testing. Add feature ablations, learning curves, shuffled-label controls and group-bootstrap intervals.

The latest uploaded `triage_classifier.py` supplied the SapBERT CLS/PCA-64 and LR/HGB baseline settings; its SHA-256 is recorded in `reports/triage_10000/environment.json`. Its preprocessing outside CV is corrected here. The original file and full private run remain local; this PR contains the maintained implementation and aggregate evidence.

## Results and interpretation

| Model on the same 1,999 test rows | Accuracy | Macro F1 |
|---|---:|---:|
| Development-selected adjusted HGB | 99.05% | 0.9888 |
| Original fused HGB baseline settings | 99.15% | 0.9901 |
| Random Forest comparison | 99.35% | 0.9926 |

The selected model did not beat the strongest baseline. Random Forest was not retrospectively selected using test scores. Earlier PDFs used different datasets and four classes, so their lower scores cannot establish a tuning gain. These results measure agreement with AI labels; strong ECG/vital-sign label patterns are discussed in the report. No clinical or level-4 performance is established.

## Reproduce training (Linux / POSIX)

Use a separate environment. The recorded run used Python 3.14.7 and the versions in `requirements.txt`; do not upgrade the application's pickle environment.

```bash
python3 -m venv .venv-study
.venv-study/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv-study/bin/pip install -r experiments/triage_study/requirements.txt
export TRIAGE_STUDY_OUTPUT="$PWD/output/results/triage_study_run"
export TRIAGE_MODEL_CACHE="$HOME/.cache/huggingface/hub"
.venv-study/bin/python experiments/triage_study/prepare_data.py /path/to/cardiac_multilingual_10000_filled.csv
.venv-study/bin/python experiments/triage_study/encode_text.py sapbert_concept
.venv-study/bin/python experiments/triage_study/encode_text.py mpnet_concept
.venv-study/bin/python experiments/triage_study/encode_text.py minilm_complaint
.venv-study/bin/python experiments/triage_study/run_study.py
```

Encoder generation is offline and requires the standard Hugging Face snapshot caches for `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, `sentence-transformers/all-mpnet-base-v2` and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Download these before an offline run. Exact original revisions and pooling settings are in `reports/triage_10000/emb_*.json`; match those cache revisions for reproduction.

Use a fresh output directory for a new dataset. Preparation refuses to overwrite another dataset or partial existing study. Once prepared, do not manually change the CSV, split assignments, encoder caches or completed configurations. The runner resumes CV caches and skips final evaluation when already complete. Research model objects refer to row-aligned embedding files; they are not live app inference models.

## Report and tests

The complete 47-page report and full-precision aggregate metrics are in `reports/triage_10000/`. Raw CSVs, record-level predictions, embeddings, cache matrices and fitted model binaries are excluded. The historical report appendices and their aggregate metrics live in `reference/`.

Rebuild the **recorded report** from committed aggregate results (this builder is specific to the recorded study):

```bash
export TRIAGE_STUDY_OUTPUT="$PWD/reports/triage_10000"
.venv-study/bin/python experiments/triage_study/build_report.py
```

Do not run training against the committed report directory. A fresh experiment has its own scores and requires updating the report's study-specific narrative and diagnostics before publication.

```bash
.venv-study/bin/python -m unittest discover -s experiments/triage_study -p 'test_*.py'
# With prepared data and embeddings, also run the actual split/PCA checks:
.venv-study/bin/python experiments/triage_study/validate_protocol.py
```

## Eight-page summary

`reports/triage_10000/Triage_Classifier_Concise_Report.pdf` is the compact report, retaining all ten final comparisons and confusion matrices, per-class scores, all five-fold refinement rows, diagnostics and historical comparison scores. Full screening configurations remain in the aggregate result files. Rebuild with `python experiments/triage_study/build_short_report.py`; the generated file is written to `output/pdf/`.

## Cache integrity and existing runs

New runs store encoder settings and the embedding array SHA-256, then freeze the prepared CSVs, split manifest and all embedding artifacts in `study_inputs.json` before training. Every matrix/CV cache read, leaderboard export and final evaluation checks this identity. Changing an encoder revision, pooling, token limit, array contents, records or splits requires a **fresh output directory**; mismatches fail without overwriting prior results.

Existing runs made before these checks lack the required fingerprints. Keep their original artifacts for reference and use a new output directory to train with this version. Do not manually add a manifest to old results. The committed historical metrics and PDFs are unchanged and can still be read or rebuilt without retraining.
