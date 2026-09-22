# Roman Urdu Medical Triage in Pakistan

An offline research prototype for cardiac complaints written in Roman Urdu.
The application translates complaints locally, checks anatomical consistency,
and combines sentence embeddings with structured patient features to predict
triage levels 1 (Emergency) through 4 (Non-Urgent).

The training data is synthetic. This system has not been clinically validated
and is not a medical device. Its classifier was trained on cardiac presentations;
broader vocabulary support does not validate it for other specialties.

## Run the application

Python 3.10 or newer and tkinter are required. Install the pinned dependencies
in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import tkinter"
```

On Windows activate with `.venv\Scripts\activate`. On Ubuntu install tkinter
with `sudo apt install python3-tk`; on Fedora use `sudo dnf install python3-tkinter`.
The saved classifier requires the pinned scikit-learn version.

Install Ollama, start its local service and obtain the translator once:

```bash
ollama serve
ollama pull qwen2.5
python triage_gui.py
```

Run `ollama serve` in a separate terminal. If an accepted translator is already
installed, the app can select it and reports its name. The sentence encoder
downloads on first use and is then cached. Prediction runs locally once these
models are present; there is no API key or cloud translation in the application.

See [HOW_TO_RUN.md](HOW_TO_RUN.md) and [the operator manual](docs/ML_Predictor_Manual.pdf).
`run_gui.sh` supports the original Linux Tk runtime layout; other installations
can use `python triage_gui.py` directly.

## Prediction pipeline

1. Normalize Roman Urdu spelling with the fuzzy clinical dictionary.
2. Translate to English using local Ollama, temperature 0.0.
3. Filter refusals and verify that named body parts survive translation.
4. Remove the English bundle's learned stop words.
5. Encode the English text with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — an SBERT (Sentence-BERT) checkpoint, the multilingual 384-dimensional variant.
6. Concatenate the L2-normalized 384-dimensional embedding with 26 structured features.
7. Predict with Logistic Regression (`class_weight="balanced"`, `max_iter=1200`).

Structured features comprise six standardized numeric inputs (age, heart rate,
systolic/diastolic pressure, temperature, SpO2) and one-hot encodings of gender,
arrival mode, AVPU and ECG status. The resulting classifier input has 410 columns.
Embeddings are not standardized again. Dictionary normalization remains a text
cleaning tool and supplies protected clinical terms for stop-word learning.

Missing/unreadable structured inputs are substituted with a warning. Translation
failures and anatomical mismatches receive no score. The existing low-signal
confidence cap remains in place. Confidence is a classifier probability, not a
validated estimate of clinical reliability.

## Desktop and batch workflows

The six tabs are Triage a Patient, Pipeline Explorer, Stop Words, Batch File,
Results and Cluster Analysis. They expose text stages, learned-word statistics,
recorded evaluation results and complaint-cluster diagnostics.

```bash
python predict_batch.py sample_100_patients.xlsx
python predict_batch.py patients.csv results.xlsx
```

Batch input columns: `Complaint_Text`, `Age`, `Gender`, `Mode_of_Arrival`,
`Heart_Rate`, `Systolic_BP`, `Diastolic_BP`, `Temperature`, `SpO2`, `AVPU`,
`ECG_Status`. CSV and XLSX are supported. Output includes triage, confidence,
notes, English translation and a per-row gate verdict. Failed rows have blank
predictions. The CLI and GUI use `triage_model_embedding_english/`.

An explicit `--model-dir` or `TRIAGE_MODEL_DIR` selects another compatible
embedding bundle for the batch CLI. A missing or incompatible bundle fails
clearly instead of silently choosing another classifier.

## Research configuration and recorded results

The active bundle is configuration C, trained on 2,252 synthetic cardiac rows
in `cardiac_english_2252.csv`. The recorded stratified 80/20 split has 1,801
training rows and 451 test rows, seed 42.

| Configuration | Accuracy | Under-triage | Over-triage | Features |
|---|---:|---:|---:|---:|
| B: raw English embeddings | 80.71% | 13.30% | 5.99% | 410 |
| C: preprocessed English embeddings (deployed) | 80.49% | 12.64% | 6.87% | 410 |

These are existing recorded results, not a new evaluation performed by the
cleanup. Preprocessing reduced recorded under-triage by 0.66 percentage points
and reduced accuracy by 0.22 points. They do not demonstrate an accuracy gain
or clinical effectiveness. The source of truth is the bundle's
`model_manifest.json` and `triage_metrics.json`.

The English training translations were prepared using gpt-4o-mini; live
translation uses Ollama. The held-out classifier evaluation therefore does
not validate the complete live translation workflow. The historical training
script also fits structured preprocessing before splitting and saves a full-data
stop-word list beside a classifier evaluated with training-only stop words.
Report these limitations; do not describe the artifact as independently validated.

## Training and embedding studies

Training is an explicit research operation; running the app does not require
retraining. Preserve the evaluated bundle and use a separate output directory
for a new experiment:

```bash
python train_embedding_pipeline.py --out-dir /tmp/triage_embedding_experiment
```

The default dataset/text column is the English subset; normalization is skipped
for English and learned stop words are removed for C. `--deploy B` selects raw
embeddings, `--deploy C` preprocessed embeddings (default), and `--deploy auto`
selects within 0.5 points of minimum under-triage before breaking ties by accuracy.
For a Roman Urdu experiment, explicitly pass `--data`, `--text-column Complaint_Text`,
`--no-skip-normalization` and a separate `--out-dir`.

`stopwords.py` learns frequent, weakly label-associated tokens using normalized
mutual information and Cramer's V while protecting clinical vocabulary.
`embedding_evaluation.py`, `check_embedding_pairs.py` and `src/cluster_analyzer.py`
study complaint similarity and nearest-neighbour cluster agreement. These are
representation diagnostics, not clinical classifier validation or text reconstruction.

## Separate professor baseline

`src/baseline.py` evaluates both `Triage_Level` and `Category` with stratified
five-fold cross-validation on the separate 185-row professor dataset.
`StaticEncoder` reads stored vectors and `DynamicEncoder` computes live vectors.
The baseline uses `intfloat/multilingual-e5-small` with the `passage: ` prefix;
these vectors cannot be exchanged with MiniLM vectors despite matching dimensions.
Benchmarks include Logistic Regression, RandomForest and HistGradientBoosting.
The saved `models_src/` classifiers are RandomForest models.

`run_inference.py` is the separate text-only baseline CLI, not the GUI's fused
cardiac classifier:

```bash
python run_inference.py --check
python run_inference.py "seena mein dard hai"
python -m src.baseline
```

## Validation and project files

```bash
python tests/audit_pipeline.py
python tests/audit_gui.py
python -m unittest discover -s tests -p 'test_*.py'
```

The audits require Ollama; the GUI audit also requires a working display.
See [DATASET_PROVENANCE.md](DATASET_PROVENANCE.md) for synthetic-data generation,
[ARCHITECTURE.md](ARCHITECTURE.md) for module responsibilities,
[FEATURES.md](FEATURES.md) for workflow details and
[SUBMISSION_SUMMARY.md](SUBMISSION_SUMMARY.md) for the article configuration.

## Article classifier and PCA experiments

See [experiments/README.md](experiments/README.md) for the separate 768-D
Sentence-BERT versus PCA-64 comparison with Logistic Regression, Random Forest
and Histogram Gradient Boosting. Metrics, LaTeX tables and figures are saved
under `output/results/`. These runs do not replace the deployed MiniLM bundle.

## Authors

- Muhammad Wasiq Hussain Siddiqui
- Abdul Mannan
- Serosh K Noon
- Muawiya Amir
- Sana Shaukat Siddiqui
- Junaid Abdullah

_Department of Biomedical Engineering - May 2026_

---

> **Disclaimer:** This is a research / educational decision-support prototype,
> not a certified medical device. It must not be used as the sole basis for
> clinical decisions. Always involve a qualified clinician.
