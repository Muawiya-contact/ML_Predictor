# Roman Urdu Medical Triage in Pakistan

An offline research prototype for cardiac complaints written in Roman Urdu.
The application translates complaints with local Ollama, verifies anatomical
consistency and combines SapBERT embeddings with patient features to predict
**0 Emergency, 1 Urgent, 2 Standard or 3 Non-urgent**.

The current model is selected using grouped cross-validation on the supplied
four-level workbook. It has not been clinically validated. Dataset-label
agreement is not a measure of clinical reliability or live translation accuracy.

## Run locally

Use Python 3.10+ with tkinter. The saved classifier requires the pinned
scikit-learn version. Install CPU PyTorch before the other dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements.txt
python -c "import tkinter"
```

On Windows use `.venv\Scripts\activate`. Install `python3-tk` on Ubuntu or
`python3-tkinter` on Fedora when tkinter is missing.

Start Ollama in a separate terminal, download Qwen and the pinned SapBERT
snapshot once, then launch the GUI:

```bash
ollama serve
ollama pull qwen2.5
hf download cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
  --revision 090663c3ae57bf35ffe4d0d468a2a88d03051a4d
python run_inference.py --check
python triage_gui.py
```

The app reads locally cached models; SapBERT inference never downloads or
substitutes another encoder. `SAPBERT_MODEL_PATH` can point to the pinned local
snapshot. `./run_gui.sh` also supports the original machine's private Tk runtime.
See [the current GUI guide](docs/SapBERT_GUI.md).

## Prediction pipeline

1. Normalize Roman Urdu spelling with the fuzzy clinical dictionary.
2. Translate to English through local Ollama, with temperature 0.0.
3. Reject failed/refused translations and anatomical mismatches.
4. Encode unfiltered English with frozen SapBERT CLS: 768 dimensions, 64-token
   limit, L2 normalization. This bundle does not remove stop words.
5. Apply the saved development-fitted PCA to obtain 128 text features.
6. Add 50 structured features: the seven numeric inputs (including ordinal AVPU)
   expand to 35 linear/quadratic terms before scaling, with 15 fitted categorical
   indicators for gender, arrival mode and ECG. Imputation is fitted on training rows.
7. Append 19 explicit details extracted from the original complaint, preserving
   duration and other mentions alongside the English SapBERT representation.
8. Classify the combined 197 features using the model named in
   `triage_model_sapbert/model_manifest.json`, selected by development CV.

GUI, `run_inference.py` and `predict_batch.py` use that same bundle. Missing or
incompatible artifacts fail clearly. `TRIAGE_MODEL_DIR` explicitly selects a
compatible alternative; it never silently changes the active model.

The six tabs are Triage a Patient, Pipeline Explorer, Stop Words, Batch File,
Results and Cluster Analysis. Results displays the active model's held-out
four-class matrix and class metrics. Cluster diagnostics use full 768-D vectors.
The Stop Words tab explains why filtering is inactive for this encoder.

When no usable complaint is entered, the GUI shows **Confidence: 50%**, no triage
level and an explanation that the value is a placeholder. It does not run the
classifier. Editing patient details clears a previous result. Failed translation
and anatomical checks also produce no level. Structured-value substitutions are
reported, and model probabilities are not calibrated clinical certainty.

## Command-line and batch use

```bash
python run_inference.py "seena mein dard hai" --age 65 --heart-rate 118
python run_inference.py --interactive
python predict_batch.py patients.csv results.xlsx
```

Batch columns: `Complaint_Text`, `Age`, `Gender`, `Mode_of_Arrival`, `Heart_Rate`,
`Systolic_BP`, `Diastolic_BP`, `Temperature`, `SpO2`, `AVPU`, `ECG_Status`.
CSV/XLSX exports preserve raw text, translation, gate verdict and reasons. Failed
rows have blank scores. The current bundle's `Predicted_Triage_Level` is 0–3.

## Current study and report

See [the improvement protocol](experiments/triage_study/IMPROVEMENT.md) for the
current search and export workflow, and [the initial protocol](experiments/triage_study/FOUR_LEVEL.md)
for source preparation and baseline comparisons.

The workbook has 10,000 rows. Its 4,290 missing concepts were recovered only after
matching all complaint/patient inputs and existing concepts to the previous
supplied file; new targets were preserved. Target names and processing metadata
are excluded from model inputs. Repeated complaint/concept groups cannot cross
holdout or CV boundaries. Patient preprocessing and PCA are fitted within folds.

The initial report retains all 12 fixed comparisons: three classifiers across
text/fused inputs and full 768-D/PCA-64 embeddings. The improvement rounds compare 55
configurations across five grouped folds (275 fits), including PCA-128/256 and
quadratic patient-feature controls. Selection uses mean macro F1 with emergency
recall no more than one percentage point below the incumbent; accuracy breaks ties.
The selected model is balanced Logistic Regression (C=100), PCA-128 plus 50
patient features, including quadratic numeric terms, and 19 original-complaint
detail features. SapBERT remains frozen.

On the previously examined 1,999-row test set, the latest improvement raises
accuracy from 86.29% to 88.99% and macro F1 from 86.80% to 89.43%. Macro precision
is 89.37%, macro recall 89.50%, and emergency recall increases from 91.42% to
93.87%. Under-triage decreases from 6.80% to 4.65%. These are retrospective
comparisons, not untouched confirmation. Five-fold development macro F1 rises
from 86.57% to 89.38%; repeated selection still requires new independent data.

The [development audit](reports/triage_learning_detail_audit/) contains nested
learning curves and the complaint-detail comparison. It flags 567 explicit
duration disagreements for review. No labels are automatically changed.
More unique, consistently labelled data may help; these curves do not predict
an accuracy at 20,000 rows.

The [current report directory](reports/triage_four_level_round4/) contains the
eight-page PDF, figures and all candidate results. The [initial comparison](reports/triage_four_level/)
remains available. Follow [the improvement workflow](experiments/triage_study/IMPROVEMENT.md)
after reproducing the initial study to recreate the active model.
The provider has no label-assignment rules available; labels remain unchanged.
Source records and individual error-review lists remain local. Saved metrics
use supplied/recovered concepts, not live Ollama translations.

The [label-task audit](reports/label_transition_audit/) explains why the older
three-level 99% scores are not directly comparable. Paired diagnostic fits hold
rows, embeddings and classifier settings fixed while changing only target labels.

## Verification and historical tools

```bash
python -m unittest discover -s tests -p 'test_*.py'
python experiments/triage_study/verify_four_level_study.py output/results/triage_four_level
```

The verifier recalculates every held-out metric and matrix, checks grouped
boundaries and CV selection, and estimates group-bootstrap uncertainty. Exporting
a bundle reproduces its entire held-out prediction vector before writing it.

Older MiniLM training, three-level studies and the separate professor baseline
remain historical research utilities, requiring their own explicitly supplied
inputs. Their encoders, labels and scores must not be combined with this study.
`src/baseline.py` and `src.offline_pipeline.run` are the separate professor
baseline; the application CLI now uses the current SapBERT bundle instead.
Historical PDFs and superseded working datasets are removed from current
report/data locations; earlier versions remain in Git history.

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
