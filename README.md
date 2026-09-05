# Domain-Guided Lightweight Feature Attention

## Hybrid Text Representation for Roman Urdu Emergency Triage Classification

> A fully offline, computationally lightweight AI triage system for emergency
> departments in Pakistan. It reads multilingual **Roman Urdu** chief complaints
> using Bag of Words + Fuzzy Matching + Diacritization, combines them with vital
> signs, and assigns a triage level. Runs on a CPU with no internet and no GPU.

---

## Run it  -  copy and paste

Open a terminal **inside the `ML_Predictor` folder** and paste one of these.
Nothing else needs setting up first; the launcher checks everything and
starts Ollama itself if it is not already running.

**Start the app**

```bash
./run_local.sh
```

Windows: `run_local.bat` (double-click it, or run it from PowerShell).

**Run the tests**

```bash
.venv/bin/python tests/audit_pipeline.py
```

```bash
.venv/bin/python tests/audit_gui.py
```

Windows: replace `.venv/bin/python` with `.venv\Scripts\python`.

**In VS Code**, with the `ML_Predictor` folder open as the workspace root:

| | |
|---|---|
| `Ctrl+Shift+B` | start the app |
| `Ctrl+Shift+P` -> `Run Task` -> *Run both test batteries* | run the tests |

The tasks are defined in `.vscode/tasks.json`. They only work when
`ML_Predictor` itself is the folder you opened - opening `Desktop` and
expanding down to it is not the same thing, and VS Code will not find them.

### What to expect

- **The first prediction takes about 11 seconds.** That is the translation
  running on your own machine. It is not a hang.
- **A 100-row batch takes roughly 18 minutes**, for the same reason - one
  local translation per row. The progress bar shows the count and an
  estimate.
- **Ollama must be running for the tests.** `./run_local.sh` starts it;
  the test commands do not. If they fail with "Ollama is not reachable",
  either run `ollama serve` in a second terminal or launch the app once
  first.
- **Close the app with the window's X, not `Ctrl+C`.** Ctrl+C interrupts
  the event loop and prints a `KeyboardInterrupt` traceback. Nothing is
  broken when that happens, but it looks alarming and it is avoidable.

---

## Table of Contents

- [Run it - copy and paste](#run-it----copy-and-paste)
- [What's New in This Version](#whats-new-in-this-version)
- [How To Run (Detailed, Beginner-Friendly)](#how-to-run-detailed-beginner-friendly)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Desktop GUI](#desktop-gui)
- [Features Guide (FEATURES.md)](FEATURES.md)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [1. Train the model](#1-train-the-model)
  - [2. Batch prediction from a file (NEW)](#2-batch-prediction-from-a-file-new-100-patients-at-a-time)
  - [3. Single / example prediction](#3-single--example-prediction)
  - [4. Interactive prediction](#4-interactive-prediction)
- [Batch Input File Format](#batch-input-file-format)
- [How It Works](#how-it-works)
- [Optional Experiment: Offline AI Embeddings vs the Dictionary](#optional-experiment-offline-ai-embeddings-vs-the-dictionary)
- [Complaint Categories Covered](#complaint-categories-covered)
- [Triage Levels](#triage-levels)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Dataset Columns](#dataset-columns)
- [Troubleshooting](#troubleshooting)
- [Authors](#authors)

---

## What's New in This Version

### Desktop GUI

[`triage_gui.py`](triage_gui.py) is a graphical interface built for
**understanding** the system, not just running it — every screen shows the
working, not only the answer.

```bash
python triage_gui.py
```

| Tab | What it shows |
|---|---|
| **Triage a Patient** | One patient → triage level, confidence bars, and the five stages the complaint passed through |
| **Pipeline Explorer** | Type any complaint and watch it move through all five stages, each rendered by calling the same function the serving path calls |
| **Stop Words** | The 68 English stop-words the serving bundle removes, with the document frequency, mutual information and Cramér's V behind each decision |
| **Batch File** | Triage a whole Excel/CSV, with a progress bar, a per-row anatomical gate verdict, and a level distribution |
| **Results** | Precision / recall / F1 / support for the serving bundle, computed from its saved confusion matrix at render time |
| **Cluster Analysis** | Pairwise similarity over a 10-complaint cluster, translated through Ollama first so it measures the path that actually ships |

Four tabs were removed in the Ollama migration: the A/B/C/D method comparison,
the embedding-effectiveness study, the model-score panel and the embedding
demo. All four scored bundles the operator can no longer reach, and accuracy
figures shown beside a live triage level read as describing it.

Uses only the Python standard library (`tkinter`), so it adds **no new
dependencies** and runs fully offline — in keeping with the project's
lightweight, CPU-only goal.

> A useful side effect: tkinter renders Unicode properly, so the diacritized
> canonical forms (`dárd`, `bukhār`, `sēna`) display correctly in the GUI even
> though the Windows console cannot print them.

**Every number in the app is read from the saved result files.** If a file has
not been generated yet, the app names the file and prints the exact command that
produces it — it never shows an invented or placeholder value.

> **New here?** [`FEATURES.md`](FEATURES.md) explains every feature in plain
> English — what it does, how it works, and why it was added.

### Embedding pipeline + two new research contributions

Implemented from [`ARCHITECTURE.md`](ARCHITECTURE.md). See that document for
the full design; this is the short version.

**Contribution 1 — automatic stop-word learner** ([`stopwords.py`](stopwords.py)).
Instead of a hand-written Roman Urdu stop-word list, the list is _learned_ from
the labelled complaints: a token is a stop word when it is common (high document
frequency) **and** uninformative (near-zero mutual information with the triage
label **and** a non-significant chi-square test). Every threshold and per-token
statistic is written to `learned_stopwords.json` so a reviewer can re-derive it.

```bash
python stopwords.py
```

Two findings worth reporting: classic filler (`hai`, `mein`, `aur`) is **not**
removed because it is statistically significant on this data; and the statistics
alone would have deleted real symptom words (`pain`, `jalan`, `khoon`), so a
clinical safety guard exempts the medical vocabulary and reports every rescue.

**Embedding pipeline** ([`train_embedding_pipeline.py`](train_embedding_pipeline.py)).
Complaint text now goes `clean → fuzzy normalize → remove learned stop words →
sentence embedding → fuse with vitals → Logistic Regression`.

```bash
pip install -r requirements-embedding.txt   # once, needs internet
python train_embedding_pipeline.py
```

Current results, copied from `embedding_pipeline_results.csv` — the app's
**Model Score** tab reads the same file live, so that is always the
authoritative view:

| Method                                | Accuracy   | Under-triage | Over-triage |                |
| ------------------------------------- | ---------- | ------------ | ----------- | -------------- |
| A) Dictionary + BoW (previous system) | 90.04%     | **3.73%**    | 6.22%       |                |
| B) Embeddings, raw text               | 91.29%     | 4.56%        | 4.15%       |                |
| C) Embeddings + preprocessing         | **92.12%** | 4.56%        | **3.32%**   | ← **deployed** |
| D) Hybrid: dictionary + embeddings    | 86.72%     | 7.05%        | 6.22%       |                |

Adding automatic stop-word removal lifted the embeddings from 91.29% → 92.12%
(**+0.83 points**) — the measured effect of Contribution 1.

**C is deployed.** Deployment is an explicit choice (`--deploy`, default `C`),
not a by-product of the scoring rule — see *How the deployed method is chosen*
below. Artifacts go to `triage_model_embedding/`, which is what
`predict_batch.py` loads by default. `triage_model/` is left untouched and is
the fallback for machines without `sentence-transformers`.

> **Note.** `triage_gui.py` no longer loads this bundle for scoring. Since the
> mode toggle was removed it serves `triage_model_embedding_english/` — see
> *Project Structure*.

**The cost of that choice, stated plainly:** C carries **+0.83 points more
under-triage** than A (4.56% vs 3.73%). Under-triage is the error that harms
patients, so this is a real trade against a 2.08-point accuracy gain and a
2.90-point over-triage improvement. `train_embedding_pipeline.py` prints this
line on every run.

### How the deployed method is chosen

Two separate things, deliberately kept apart:

1. **The safety-first rule** — treat every method whose under-triage is within
   0.5 points of the lowest as safety-equivalent, then take the most accurate.
   This is still computed and reported on every run.
2. **What actually ships** — the `--deploy` flag, default `C`.

They were the same thing until it emerged that the rule alone could not decide
this: when two methods tie, `max()` keeps the first, and the first is the
dictionary baseline. So the "embedding pipeline" shipped a dictionary-only
model with `embedding_model: null` in its own metrics. Use `--deploy auto` to
fall back to the old rule-decides-everything behaviour.

> **Fixed — the hybrid was not contributing.** Method D used to report numbers
> identical to method A on all three metrics (90.04 / 3.73 / 6.22). The
> attention-weighted Bag-of-Words block and the L2-normalised embedding block
> were concatenated without rescaling, so under a shared L2 penalty the
> embedding columns cost far more coefficient per unit of signal and the
> optimiser dropped all 384 of them. Each block is now standardised separately
> (fit on the training split only) before concatenation, and
> `train_embedding_pipeline.py` prints both blocks' scales so the fault cannot
> return silently. **Note the honest outcome:** properly fused, D is *worse*
> than either input alone (86.72%, 7.05% under-triage) — 1,094 features on 963
> training rows. Fixing the fusion was necessary to know that; it did not
> produce a better model, which is why C is deployed.

**Contribution 2 — embedding-effectiveness study**
([`embedding_evaluation.py`](embedding_evaluation.py)). Measures how faithfully
the embedding generator represents complaints. Does not touch the classifier.

```bash
python embedding_evaluation.py
```

10 manual meaning-clusters ([`evaluation_clusters.json`](evaluation_clusters.json),
editable), 441 pairwise comparisons (45 per 10-complaint cluster), each with
cosine similarity, Pearson correlation and Euclidean distance.

| Measure                                          | Value           |
| ------------------------------------------------ | --------------- |
| Mean within-cluster similarity                   | 0.477           |
| Mean across-cluster similarity                   | 0.317           |
| Separation gap                                   | +0.159          |
| Pairs above the 0.5 threshold                    | 209/441 (47.4%) |
| **Embedding generator efficiency** (closed pool) | **71.7%**       |
| Embedding generator efficiency (open pool)       | 39.4%           |

The "regenerate the text from the embedding" test is implemented as
**nearest-neighbour fidelity**: sentence embeddings are lossy and one-way, so
literal inversion is not possible (see `ARCHITECTURE.md` Task 4c). Instead each
complaint's closest neighbour is checked for same-meaning membership.

> **Honest finding.** Roman Urdu clusters only weakly, exactly as
> `ARCHITECTURE.md` §4.1 predicted. Per-cluster results range from
> unconsciousness (82% of pairs above threshold) down to breathing difficulty
> (24%). `seena mein shadeed dard aur pasina aa raha hai` (chest pain) scores
> 0.94 against `saans lene mein takleef wheezing ho rahi hai` (breathing) — the
> model is partly encoding "Roman Urdu sentence-ness" rather than meaning. This
> is evidence for transliterating to native script or fine-tuning, and is
> reported rather than tuned away.

---

Two features were added on top of the original system.

**1. Batch prediction from an uploaded file (`predict_batch.py`).**
You can now triage many patients at once (e.g. 100 in one go) by giving the
program a single **Excel (`.xlsx`)** or **CSV** file. It writes a results file
with the predicted triage level, label, confidence, and the full probability
breakdown for every patient, and prints a summary to the screen. Missing values
and unknown category labels are handled gracefully instead of crashing.

**2. Wider complaint coverage (more complaints the model is trained on).**
The original symptom vocabulary was almost entirely cardiac. The recognition
dictionaries (diacritization map, fuzzy vocabulary, and clinical attention
weights) were expanded to also cover **trauma, burns, infectious disease,
neurological emergencies (seizure / stroke), gyne / obstetric, metabolic
(diabetic), heat-related, allergy / skin, and psychiatric** complaints. The
model is then retrained so all of these complaint types are actually learned
instead of being treated as noise.

**Supporting change.** The text pipeline used to be copy-pasted into three
files that had already started to disagree. It now lives in one shared module,
[`triage_pipeline.py`](triage_pipeline.py), which training and every predictor
import. Training and inference can no longer drift apart.

> Note: "widening the vocabulary" improves how many complaint _types_ the model
> can read from the existing dataset. The single biggest further improvement
> would be adding more **real labelled patient records** for the non-cardiac
> categories, which are still under-represented in the dataset.

---

## Overview

Patients in Pakistani EDs describe symptoms in **Roman Urdu** (Urdu written in
the English alphabet) with no standard spelling. _Fever_ alone can appear as
`bukhar`, `bukhaar`, `bukhr`, `bkhar`, or `garmi`.

The system now has **one** triage path, and every stage of it runs on the
machine in front of you:

```
raw Roman Urdu
   │  1. dictionary + fuzzy normalization      local, deterministic, logged
   ▼
canonical Roman Urdu
   │  2. Ollama on localhost                   llama3.2, temperature 0
   ▼
clinical English
   │  3. anatomical assertion gate             deterministic; blocks or passes
   ▼
   │  4. stop-word removal + sentence-transformer
   ▼
384-dim vector  →  RandomForest  →  triage level 1-4 + confidence
```

No stage reaches the internet. Ollama is a service on `127.0.0.1`, the encoder
loads from the on-disk Hugging Face cache, and the classifiers are pickles in
the model bundle.

**The anatomical gate is the safety check, not cosine similarity.** Every body
part named in the complaint must survive into the English translation, or the
prediction is refused. Cosine is still measured and printed, and it decides
nothing: against `seena mein shadeed dard aur pasina`, a correct translation
scores 0.8054 and *"My leg is broken after a fall"* scores 0.7922 — 0.013
apart, with the badly drifted candidate outscoring the mildly drifted one. No
threshold separates those. The gate does, exactly, for the one failure mode
that matters here.

**What the gate cannot do:** it checks anatomy, not fidelity. A wrong *symptom*
attached to the right body part passes.

---

## Project Structure

```
ML_Predictor/
|
|-- triage_gui.py                   # desktop GUI, 6 tabs (tkinter, no extra dependencies)
|-- triage_pipeline.py              # SHARED pipeline: dictionaries, normalize, features, predict
|-- stopwords.py                    # automatic stop-word learner (Contribution 1)
|-- train_embedding_pipeline.py     # embedding -> fuse -> classify training flow
|-- predict_batch.py                # batch prediction from an Excel/CSV file
|-- generate_cardiac_dataset.py     # reviewable synthetic dataset generator
|-- embedding_evaluation.py         # embedding-effectiveness study (Contribution 2)
|-- check_embedding_pairs.py        # verifies the similarity figures quoted in SUBMISSION_SUMMARY
|
|-- src/                            # the offline Ollama pipeline
|   |-- offline_pipeline.py         # prompt, translation, guardrail, fuzzy dictionary, anatomical gate
|   |-- embedding_pipeline.py       # translate -> normalize -> 384-dim vector
|   |-- cluster_analyzer.py         # pairwise similarity over a cluster of complaints
|   |-- train.py / predict.py       # the 185-row professor-baseline reproduction
|   |-- README.md
|-- run_inference.py                # CLI: Roman Urdu in, triage + department out
|-- Modelfile                       # optional custom Ollama model (med-translator)
|
|-- tests/
|   |-- audit_pipeline.py           # 15 checks: prompt, guardrail, gate, fuzzy, skew, live translations
|   |-- audit_gui.py                # 16 checks: builds the real window and drives every tab
|   |-- fixtures/batch_sample.csv   # 8 rows chosen for edge cases, not the happy path
|
|-- learned_stopwords.json          # learned list + per-token statistics
|-- evaluation_clusters.json        # 10 manual meaning-clusters (editable)
|-- cardiac_multilingual_10000_v3.csv    # deployed training data (SYNTHETIC - see DATASET_PROVENANCE.md)
|-- batch_input_template.xlsx/.csv  # blank template to fill in your patients
|
|-- triage_model_embedding/         # 10,000-row Roman Urdu bundle
|-- triage_model_embedding_english/ # 2,252-row English bundle  <- THIS IS WHAT THE GUI SERVES
|-- models_src/                     # 185-row professor-baseline bundle (used by run_inference.py)
|
|-- docs/architecture_flow.html     # rendered architecture walkthrough
|-- professor_baseline/             # the original baseline scripts, kept as-is for provenance
```

> **Which bundle serves?** `triage_model_embedding_english/`. The mode toggle
> was removed, and with it GUI access to the 10,000-row Roman Urdu bundle —
> that bundle expects Roman Urdu and cannot be fed translated English without
> train/serve skew. The GUI therefore scores with 2,252 rows, not 10,000. This
> is deliberate and it is a real reduction in training data.


---

## Installation

**Requirements:** Python 3.8+

```bash
# 1. Open a terminal inside the ED/ folder

# 2. (Optional but recommended) create a virtual environment
python -m venv venv
# Windows:  venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

Windows users: two helper PowerShell scripts are included to simplify setup and running:

- `setup_env.ps1` — creates a `.venv` virtual environment and installs `requirements.txt`.
- `run_predict.ps1` — activates the `.venv` (if present) and runs `predict_batch.py` (accepts input/output args).

Run them from PowerShell inside the project folder:

```powershell
# create venv and install deps
.\setup_env.ps1

# run the example batch prediction
.\run_predict.ps1
```

`requirements.txt`:

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn==1.6.1  # pinned: the saved model was built with this version
scipy>=1.9.0         # stopwords.py imports scipy.stats directly
rapidfuzz>=3.0.0
joblib>=1.2.0
matplotlib>=3.6.0
openpyxl>=3.1.0      # needed for reading/writing Excel (.xlsx) files
```

`requirements-embedding.txt` (needed for the embedding pipeline, the embedding
evaluation, and the app's **Embedding Demo**):

```
sentence-transformers>=3.0.0
```

The GUI itself needs no extra install — it uses `tkinter`, which ships with
Python.

---

## Quick Start

A trained model is already included, so you can predict immediately.

**Easiest — open the desktop app:**

```bash
python triage_gui.py
```

**Or from the command line:**

```bash
# Triage the 100 example patients and write a results file
python predict_batch.py sample_100_patients.xlsx
```

To retrain from scratch first (optional):

```bash
python train_embedding_pipeline.py
```

> **New to Python / sharing this with non-technical friends?**
> See **[`HOW_TO_RUN.md`](HOW_TO_RUN.md)** — a separate, very simple step-by-step
> guide written for someone who has never used Python or a terminal. The detailed
> beginner walkthrough below covers the same ground.

---

## How To Run (Detailed, Beginner-Friendly)

**Never used Python before? Start here.** Follow the steps in order.
**Steps 1 to 4 are done once per computer.** After that you only repeat Step 5.

Every command below has been run and checked on this project. You can copy and
paste each one exactly as it is written.

---

### Step 1 — Install Python (one time)

1. Go to **https://www.python.org/downloads/**
2. Click the big **Download Python** button.
3. Open the file you just downloaded.
4. **Tick the box that says "Add Python to PATH".** This is on the first screen
   of the installer. It is very important — the commands will not work without it.
5. Click **Install Now** and wait for it to finish.
6. To check it worked, open **PowerShell** (click Start, type `powershell`,
   press Enter) and type:

   ```bash
   python --version
   ```

   You should see something like `Python 3.12.0`. If you see an error, install
   Python again and make sure you tick **Add Python to PATH**.

---

### Step 2 — Get the project folder onto your computer

1. If you have a `.zip` file, right-click it → **Extract All** → **Extract**.
2. Open the folder until you can see the file `triage_gui.py` inside it.
   **That folder is the one you work in.**

---

### Step 3 — Open a terminal inside the project folder

1. Open the folder that has `triage_gui.py` in it.
2. Click the **address bar** at the top of the window (where the folder path is
   shown).
3. Type `powershell` and press **Enter**.

A black or blue window opens. It is already pointing at your project folder.
**Use this window for every command below.**

---

### Step 4 — Install what the program needs (one time)

Type this and press Enter. It downloads the basic parts:

```bash
pip install -r requirements.txt
```

Then type this and press Enter. It downloads the AI embedding part:

```bash
pip install -r requirements-embedding.txt
```

Both are one-time downloads and need internet. **After this the program works
offline** — you do not need internet again.

> The very first time you run an embedding command, it also downloads the AI
> model (a few hundred MB). That is also one time only. Everything after that
> runs with no internet.

---

### Step 5 — Open the app (this is the part you repeat)

```bash
python triage_gui.py
```

A window opens. This is the easiest way to use the whole project — you can
triage a patient, run a whole file of patients, and look at the results and
scores, all with buttons instead of commands.

---

### Step 6 — Use your own patients

1. Open `batch_input_template.xlsx`, then **File → Save As** →
   `my_patients.xlsx`. Save it in the **same folder**.
2. Keep the top row (the headings). Replace the example rows with your own
   patients, one patient per row. Only `Complaint_Text` is required; the other
   columns are optional.
3. In the app, go to the **Batch File** tab, click **Browse...**, choose your
   file, and click **Run batch triage**.

Prefer the command line? This does the same thing:

```bash
python predict_batch.py my_patients.xlsx
```

It creates `my_patients_predictions.xlsx` and `my_patients_predictions.csv`.

> The file name in the command must match your file **exactly**, and the file
> must be in the same folder. Type `dir` and press Enter to list the files in
> the folder if you are not sure.

---

## All Commands (copy and paste)

Run every command from inside the project folder — the one containing
`triage_gui.py`. Each command below has been tested and works.

### Everyday use

| What you want to do | Command |
|---|---|
| **Open the app (easiest)** | `python triage_gui.py` |
| Triage a file of patients | `python predict_batch.py sample_100_patients.xlsx` |
| Triage your own file | `python predict_batch.py my_patients.xlsx` |
| Choose where to save results | `python predict_batch.py my_patients.xlsx my_results.xlsx` |
| Triage one complaint from the terminal | `python run_inference.py "seena mein dard hai"` |
| Check Ollama, the encoder and the classifiers | `python run_inference.py --check` |
| Force the dictionary model instead of the deployed one | `python predict_batch.py my_patients.xlsx --model-dir triage_model` |

**Which model do these use?** `predict_batch.py` loads
`triage_model_embedding/` and falls back to `triage_model/` if
`sentence-transformers` is missing. `triage_gui.py` serves
`triage_model_embedding_english/`. `run_inference.py` uses `models_src/`, the
185-row professor-baseline bundle, and also predicts a department.

Three bundles, three consumers — and **two of the encoders both emit 384
dimensions**, so a vector from one will load into the other, predict, and be
wrong with nothing to signal it. Every bundle therefore reads its encoder name
from its own `manifest.json` rather than a constant. Each entry point prints
the method it is using at startup.

CSV files are read with encoding detection (`utf-8` → `utf-8-sig` → `cp1252` →
`latin-1`), so a sheet exported from Excel on Windows loads instead of failing
with `UnicodeDecodeError`.

### Training and research

These take longer to run. You only need them if you change the data or the
dictionaries, or if you are writing up the results.

| What you want to do | Command |
|---|---|
| Learn the stop-word list on its own | `python stopwords.py` |
| **Train + deploy the embedding model** | `python train_embedding_pipeline.py` |
| Deploy a different method instead | `python train_embedding_pipeline.py --deploy D` |
| Let the safety rule decide (old behaviour) | `python train_embedding_pipeline.py --deploy auto` |
| Measure how good the embeddings are | `python embedding_evaluation.py` |
| Run the pipeline test battery (15 checks) | `python tests/audit_pipeline.py` |
| Run the GUI test battery (16 checks) | `python tests/audit_gui.py` |

Both batteries need a live Ollama. There is no CI — they are run by hand.

**What each one writes:**

- `stopwords.py` → updates `learned_stopwords.json`
- `train_embedding_pipeline.py` → updates `triage_model_embedding/` (including
  `model_manifest.json`, which is how every predictor knows what that directory
  holds), `embedding_pipeline_results.csv` and `learned_stopwords.json`
- `embedding_evaluation.py` → updates `embedding_evaluation_results.csv`,
  `embedding_evaluation_pairs.csv` and `embedding_evaluation_neighbours.csv`

> **Order matters for the app's Results tab.** Run
> `python train_embedding_pipeline.py` and then `python embedding_evaluation.py`
> at least once. Until you do, the **Results** and **Model Score** tabs will tell
> you which command to run instead of showing a number. They never show a made-up
> value.

### Windows helper scripts (optional)

Two PowerShell helpers are included if you prefer them:

```powershell
.\setup_env.ps1     # creates a .venv folder and installs requirements.txt
.\run_predict.ps1   # runs predict_batch.py using that .venv
```

---

## Usage

> Run every command from inside the `ED/` folder.

### 1. Train the model

```bash
python train_embedding_pipeline.py
```

Reads the deployed dataset, runs the full text pipeline, trains the classifier,
prints accuracy + safety metrics, and saves to `triage_model_embedding/`.
**Run this whenever you change any dictionary in `triage_pipeline.py`.**

### 1b. Start Ollama (required for the GUI and `run_inference.py`)

```bash
ollama serve
ollama pull llama3.2
```

`ollama serve` does not survive a reboot on its own. If it keeps stopping:

```bash
sudo systemctl enable --now ollama
```

Check everything the pipeline needs, without running an inference:

```bash
python run_inference.py --check
```

### 2. Batch prediction from a file (NEW, 100 patients at a time)

This is the new feature for triaging many patients from one uploaded file.

```bash
# Use the default template
python predict_batch.py

# Use your own Excel file
python predict_batch.py my_patients.xlsx

# Use your own CSV file
python predict_batch.py my_patients.csv

# Choose where to save the result
python predict_batch.py my_patients.xlsx my_results.xlsx
```

What it does:

1. Reads every patient row from your `.xlsx` or `.csv` file.
2. Runs each one through the same pipeline used in training.
3. Saves `<yourfile>_predictions.xlsx` **and** `<yourfile>_predictions.csv`
   with these extra columns added to your original data:
   `Predicted_Triage_Level` (1-4), `Predicted_Label`, `Confidence`,
   `P_L0 / P_L1 / P_L2 / P_L3`, and `Notes`.
4. Prints a count of how many patients fell into each triage level, plus a
   preview of the first rows.

There is no hard limit on the number of patients (100, 500, 1000+ all work).

**To try it right now:**

```bash
python predict_batch.py sample_100_patients.xlsx
```

### 3. Triage one complaint from the terminal

```bash
python run_inference.py "seena mein dard kandhay tak ja raha hai"
```

Prints the resolved Ollama model, the translation, predictions from BOTH the
Roman Urdu and English vectors, the cosine similarity (diagnostic only), and
the anatomical gate verdict — which is what decides the accepted answer.

### 4. Interactive mode

```bash
python run_inference.py --interactive
```

---

## Batch Input File Format

Your Excel/CSV file needs a **header row** with these columns (order does not
matter). Open `batch_input_template.xlsx` to see a filled-in example.

| Column            | Example value                          | Notes                          |
| ----------------- | -------------------------------------- | ------------------------------ |
| `Complaint_Text`  | `seena mein dard aur pasina`           | Roman Urdu / English free text |
| `Age`             | `62`                                   | years                          |
| `Gender`          | `Male` / `Female`                      |                                |
| `Mode_of_Arrival` | `Ambulance` / `Walk-in` / `Wheelchair` |                                |
| `Heart_Rate`      | `120`                                  | bpm                            |
| `Systolic_BP`     | `158`                                  | mmHg                           |
| `Diastolic_BP`    | `96`                                   | mmHg                           |
| `Temperature`     | `37.1`                                 | Celsius                        |
| `SpO2`            | `93`                                   | %                              |
| `AVPU`            | `A` / `V` / `P`                        | consciousness                  |
| `ECG_Status`      | `ST elevation`, `Normal`, ...          | see dataset for full list      |

**Robustness.** If a number is missing it is filled with the training average,
and if a category value was never seen during training it falls back to a known
value. Either way the row still gets a prediction, and the substitution is
recorded in that row's `Notes` column so you can review it.

---

## How It Works

The pipeline has five stages (all defined once in `triage_pipeline.py`).

### 1. Text normalization

Lowercases, removes punctuation, and applies fast high-confidence rule-based
replacements (e.g. `seenay` -> `seena`).

### 2. Fuzzy matching

[RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) `token_sort_ratio` catches
spelling variants not covered by rules. Any word scoring >= 80% against the
canonical medical vocabulary is replaced with the canonical form
(`bukhr` -> `bukhar`, `chakar` -> `chakkar`).

### 3. Diacritization

A dictionary maps every known spelling variant to a single diacritized
canonical form, so all variants collapse to one token
(`dard, dardh, durd, drd` -> `dárd`).

### 4. Hybrid Bag of Words

Two `CountVectorizer`s run together:

| Vectorizer | Analyzer | n-gram | Features | Purpose                           |
| ---------- | -------- | ------ | -------- | --------------------------------- |
| Word BoW   | word     | (1, 2) | 350      | medical terms + bigrams           |
| Char BoW   | char_wb  | (2, 4) | 350      | sub-word patterns, typo tolerance |

Concatenated into one 700-feature text matrix.

### 5. Domain-guided lightweight feature attention

Each text feature is multiplied by a clinically motivated weight. Critical terms
are boosted (`arrest` 3.2, `behosh` 3.0, `stroke` 3.0, `seena` 2.8); filler
words are suppressed (`hai` 0.6). The weighted text features are concatenated
with the structured features (age, vitals, gender, arrival mode, AVPU, ECG) and
fed into a Logistic Regression classifier.

---

## Optional Experiment: Offline AI Embeddings vs the Dictionary

A hand-built dictionary (fuzzy match + diacritization) can never catch _every_
spelling of every symptom. A common alternative is to let an **offline AI
embedding model** turn each complaint into a "meaning vector," so that
`dard`, `drd`, and `pain` land close together automatically — no dictionary
entry needed.

`embedding_evaluation.py` measures whether that actually helps **on our own
data**, instead of assuming it does — see *Contribution 2* above.

> The older `embedding_experiment.py` comparison was removed along with the
> GUI panels that displayed it. Both described bundles the operator can no
> longer reach. The measurements it produced are still in
> `embedding_evaluation_results.csv` and in git history.

> **Honest expectation.** These embedding models are trained mostly on
> _native-script_ Urdu (اردو) and English, so _Roman_ (Latin-script) Urdu is
> under-represented. The synonym test may show weak clustering. That's exactly
> why this is an experiment: if embeddings (B or C) beat the dictionary (A) on
> our data, adopt them; if not, we keep the dictionary and have real numbers to
> show for it. A middle path — transliterating Roman Urdu to native script
> before embedding — is a further step if B/C look promising but imperfect.
>
> **Note on tooling.** This uses `sentence-transformers`, which runs offline in
> Python and plugs straight into our classifier. A phone chat app like PocketPal
> proves offline AI is practical, but it is a chat UI with no code interface, so
> it cannot be wired into this pipeline. For a _programmable_ offline LLM (e.g.
> to translate complaints first), Ollama or llama-cpp-python on a laptop are the
> integrable options — heavier, and worth weighing against the project's
> lightweight, CPU-only goal.

---

## Complaint Categories Covered

The expanded vocabulary now recognizes terms across these complaint families:

| Category         | Example recognized terms (Roman Urdu / English)                              |
| ---------------- | ---------------------------------------------------------------------------- |
| Cardiac          | seena, dard, dhadkan, palpitation, pasina                                    |
| Respiratory      | saans, phoolna, breath, dyspnea, wheezing                                    |
| Neurological     | behosh, chakkar, **daura / mirgi / seizure**, **falij / stroke / lakwa**     |
| Trauma           | **chot / zakhm / injury**, **haddi / fracture**, **accident**, moch / sprain |
| Burns            | **jalna, jhulas, scald, burn**                                               |
| Infectious / GI  | bukhar, ulti, **dast / diarrhea**, **infection / sepsis**                    |
| Metabolic        | **sugar / diabetes / hypoglycemia**                                          |
| Heat-related     | **heatstroke, dehydration**                                                  |
| Gyne / Obstetric | **haml / pregnancy, delivery, labour, miscarriage**                          |
| Allergy / Skin   | **kharish / itching, rash, allergy, reaction**                               |
| Psychiatric      | **ghabrahat / anxiety, panic, bechaini**                                     |

(Bold = added in this version.)

---

## Triage Levels

The dataset labels patients `1-4`; the model works internally with classes
`0-3`. The mapping is:

| Dataset level | Model class | Label      | Meaning                                |
| ------------- | ----------- | ---------- | -------------------------------------- |
| 1             | 0           | EMERGENCY  | Immediate attention - life-threatening |
| 2             | 1           | URGENT     | Seen within 15 minutes                 |
| 3             | 2           | STANDARD   | Seen within 60 minutes                 |
| 4             | 3           | NON-URGENT | Can wait or be redirected              |

In the batch output, `Predicted_Triage_Level` is reported on the 1-4 scale to
match the dataset.

---

## Model Architecture

```
Raw Roman Urdu Complaint
        |
        v
 Text Normalization  (lowercase, punctuation, rule replacements)
        |
        v
 Fuzzy Matching      (RapidFuzz, threshold = 80%)
        |
        v
 Diacritization      (variant -> canonical phonetic form)
        |
        v
 Hybrid BoW (700)    [ Word BoW 350 | Char BoW 350 ]
        |
        v
 Domain-Guided Attention   (multiply by clinical weights 0.5 - 3.2)
        |
 + Structured Features (9): Age, HR, Systolic, Diastolic, Temp, SpO2,
        |                   Gender, Mode of Arrival, AVPU, ECG
        v
 Combined Feature Vector (~709 features)
        |
        v
 Logistic Regression  (class_weight='balanced', max_iter=1200)
        |
        v
 Triage Level: 0 / 1 / 2 / 3
```

**Model size:** < 3,000 parameters, 5-10 MB on disk. Fully offline, CPU only.

---

## Evaluation Metrics

Two models are scored in this project. They are **separate pipelines, not two
views of one model**, so their numbers are never combined:

| | Deployed — `triage_model_embedding/` | Fallback — `triage_model/` |
| --- | --- | --- |
| Method | C) Embeddings + preprocessing | Dictionary + Bag-of-Words |
| Text features | sentence-transformer, 384 dims | word/char n-gram counts × attention |
| Accuracy | **92.12%** | ~90.5% |
| Under-triage | 4.56% | ~4.1% |
| Over-triage | 3.32% | ~5.4% |
| Safety grade | A+ | A+ |
| Metrics file | `triage_model_embedding/triage_metrics.json` | `triage_model/triage_metrics.json` |

Both measured on the same 20% held-out test set (241 patients). The **Model
Score** tab reads both files live and labels each card with the pipeline that
produced it.

Beyond accuracy, two clinical safety metrics matter most:

- **Under-triage** - patient rated _less_ urgent than they should be. Dangerous.
- **Over-triage** - patient rated _more_ urgent than needed. Wastes resources.

**Safety grading (under-triage):** A+ <5%, A <10%, B <15%, C <20%, F >=20% (do not deploy).
**Efficiency grading (over-triage):** A <15%, B <25%, C <35%, D >=35%.

Exact numbers are written to the relevant `triage_metrics.json` each time you
train — `triage_model/` by the archived dictionary trainer, and
`triage_model_embedding/` by `train_embedding_pipeline.py`.

---

## Dataset Columns

| Column            | Type         | Description                                 |
| ----------------- | ------------ | ------------------------------------------- |
| `Complaint_Text`  | Text         | Free-text complaint in Roman Urdu / English |
| `Age`             | Numeric      | Patient age (years)                         |
| `Gender`          | Categorical  | Male / Female                               |
| `Mode_of_Arrival` | Categorical  | Ambulance / Walk-in / Wheelchair            |
| `Heart_Rate`      | Numeric      | bpm                                         |
| `Systolic_BP`     | Numeric      | mmHg                                        |
| `Diastolic_BP`    | Numeric      | mmHg                                        |
| `Temperature`     | Numeric      | Celsius                                     |
| `SpO2`            | Numeric      | Oxygen saturation %                         |
| `AVPU`            | Categorical  | Alert / Voice / Pain / Unresponsive         |
| `ECG_Status`      | Categorical  | Normal / ST elevation / Arrhythmia / etc.   |
| `Triage_Level`    | Target (1-4) | Ground-truth triage level                   |
| `Category`        | Label        | Cardiac / Respiratory / Trauma / etc.       |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'openpyxl'`**
Run `pip install -r requirements.txt`. `openpyxl` is required to read/write Excel files.

**`Input file not found`**
Pass the correct path, e.g. `python predict_batch.py sample_100_patients.xlsx`,
and run the command from inside the `ED/` folder.

**`FileNotFoundError: triage_model/model.pkl`**
Train first: `python train_embedding_pipeline.py`. The app needs only *one* of the
two bundles to start, so `python train_embedding_pipeline.py` also works.

**`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfb...` on a CSV**
Fixed — CSV reading now detects the encoding. If you still see this, you are on
an old copy: the shared reader is `read_table()` in `triage_pipeline.py`, used by
both `predict_batch.py` and the app's Batch File tab.

**The app says it is using the dictionary model, not embeddings**
`sentence-transformers` is missing, so it fell back on purpose. The status bar
says so. Install it with `pip install -r requirements-embedding.txt`, or run
`python train_embedding_pipeline.py` if `triage_model_embedding/` is absent.

**`ModuleNotFoundError: No module named 'tkinter'` (Linux)**
Some distributions ship tkinter separately from Python. On Fedora/RHEL:
`sudo dnf install python3-tkinter`; on Debian/Ubuntu: `sudo apt install python3-tk`.
It is part of the standard library, so there is no `pip` package for it.

**Predictions look off after editing a dictionary in `triage_pipeline.py`**
You must retrain (`python train_embedding_pipeline.py`) so the saved model and the
attention weights match the new vocabulary.

**Rows have text in the `Notes` column**
That row had a missing number or an unknown category value that was auto-filled.
The prediction is still produced; review the note if accuracy matters for that row.

**`InconsistentVersionWarning` or `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`**
The saved model in `triage_model/` was built with **scikit-learn 1.6.1** (pinned in
`requirements.txt`). A different installed version can fail to load the pickle.
Fix by matching the version: `pip install scikit-learn==1.6.1`. If you prefer a
newer scikit-learn, just retrain once with it (`python train_embedding_pipeline.py`),
which regenerates the model files against your installed version.

Note that **1.6.1 has no wheel for Python 3.14** and needs a C compiler to build
from source. On Python 3.13 or newer, install the current scikit-learn and
retrain both bundles rather than fighting the pin:

```bash
pip install scikit-learn
python train_embedding_pipeline.py     # regenerates triage_model_embedding/
python train_embedding_pipeline.py     # regenerates triage_model_embedding/
```

A pickle written by a newer scikit-learn generally will **not** load on an older
one, so if you hand this project to someone else, either ship the version you
trained with or tell them to retrain.

---

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
