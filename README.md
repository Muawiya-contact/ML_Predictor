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
4. Encode English + `[SEP]` + the original complaint with frozen SapBERT CLS: 768 dimensions, 128-token
   limit, L2 normalization. This bundle does not remove stop words.
5. Apply the saved development-fitted PCA to obtain 64 text features.
6. Add 50 structured features: the seven numeric inputs (including ordinal AVPU)
   expand to 35 linear/quadratic terms before scaling, with 15 fitted categorical
   indicators for gender, arrival mode and ECG. Imputation is fitted on training rows.
7. Append 19 explicit details extracted from the original complaint, preserving
   duration and other mentions alongside the English SapBERT representation.
8. Classify the combined 133 features using the model named in
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

The current model is frozen SapBERT with paired text input, PCA-64, 50 patient
features and 19 original-complaint details: 133 inputs to balanced Logistic
Regression C=100. The encoder receives the supplied clinical concept plus
`[SEP]` plus the original complaint during the study, with a 128-token limit.
The GUI uses locally translated English plus the same original complaint.

All 10,000 supplied rows and labels 0/1/2/3 are unchanged. The original 8,001
 development / 1,999 test partition and five grouped folds are retained.
The initial concept recovery matched all complaint/patient fields before
copying missing concepts. Target names and processing metadata are excluded.
PCA, imputation and scaling fit within training folds.

The [advanced comparison](experiments/triage_study/ADVANCED_SEARCH.md) tested
ordered classifiers, cubic numeric features, neural classifier heads, paired
complaint input and fixed probability combinations. Across all rounds there
are 133 configurations: 109 trainable settings (545 grouped-fold fits) and
24 blends (120 reused-probability evaluations). The new work adds 30 trainable
settings and 24 blends to the previous comparison. SapBERT is not fine-tuned.

Selection uses mean development macro F1 with emergency recall no more than
one percentage point below the original reference. The paired-text LR achieved
90.49% development accuracy and 90.83% macro F1, ahead of the best probability
ensemble. On the previously examined test records:

| Metric | Previous model | Selected paired-text model |
| --- | ---: | ---: |
| Accuracy | 89.29% | 90.60% |
| Macro precision | 89.67% | 90.98% |
| Macro recall | 89.81% | 91.00% |
| Macro F1 | 89.73% | 90.99% |
| Emergency recall | 95.10% | 94.36% |
| Under-triage | 4.65% | 4.20% |

All four requested aggregate metrics exceed 90% in these comparisons.
Individual class scores do not all exceed 90%. Emergency recall decreases
slightly on the old test records, while aggregate accuracy/F1 improve. These
are retrospective results, not untouched independent or clinical validation.
The conditional paired development F1 gain interval versus the prior model is
+0.70 to +1.67 percentage points and excludes selection uncertainty.

Use the [selected-model paper package](reports/triage_selected_model/) for the
current model report, general project workflow, and captioned figures. The
[full search archive](reports/triage_four_level_round7/) retains all settings,
original full-768/PCA-64 baselines, nine pipeline representatives and audits. Source records, individual error queues
and OOF arrays stay local. The provider has no label-assignment rules available;
no labels were changed. Live translation accuracy is not measured here.

The [learning/detail audit](reports/triage_learning_detail_audit/) and
[label-task audit](reports/label_transition_audit/) remain historical evidence.
Older three-level scores concern a different target; they are not directly
comparable to this four-level experiment.

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

The general guide is [Roman Urdu Triage Project Workflow](reports/triage_selected_model/Roman_Urdu_Triage_Project_Workflow.pdf).
The [selected-model report](reports/triage_selected_model/SapBERT_Final_Paper_Report.pdf) contains measured model results without historical classifier tables.
The [complaint-mention diagnostic](reports/complaint_semantic_audit/) uses four fixed 100-record development samples; its rule-derived groups and similarity scores are not triage accuracy.

The [previous-versus-latest comparison](reports/triage_historical_comparison/) explains the changed labels, matrix counts and score differences in a separate companion PDF.
