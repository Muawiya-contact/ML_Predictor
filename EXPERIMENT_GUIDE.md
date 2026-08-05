# Embedding Experiment — Simple Guide

Hi! This short guide explains **what this experiment is** and **how to run it**,
step by step. You don't need to know machine learning. Just follow along.

If you only want the run steps, jump to **"How to run it"** below.

---

## What is this, in plain words?

We built an AI system that reads a patient's complaint (written in Roman Urdu,
like `seena mein dard aur pasina`) plus their vitals, and decides how urgent
they are — **Level 1 (Emergency)** down to **Level 4 (Non-urgent)**.

Right now the system reads the text using a **dictionary** we built by hand
(it knows `dard`, `drd`, `dardh` all mean "pain", etc.). The problem: a
dictionary always misses some spellings.

This experiment tests a different idea: use a small **offline AI model** that
understands *meaning*, so `dard`, `drd`, and `pain` are treated as similar
automatically — no dictionary needed. The script checks whether that AI model
makes our triage **more accurate**, less accurate, or about the same, using our
own patient data.

It compares three ways of reading the text and tells you which wins:

- **A. Dictionary** (what we use now)
- **B. AI embeddings only** (the new idea)
- **C. Both together**

**Important:** we are NOT building a new AI from scratch. We just download a
ready-made small model and test if it helps. Our own triage decision-maker
stays the same.

---

## What you need before starting

1. A **laptop/PC** (Windows, Mac, or Linux). A phone won't work for this.
2. **Python** installed. To check: open a terminal / PowerShell and type
   `python --version`. If you see something like `Python 3.12`, you're good.
   If not, install it from **https://www.python.org/downloads/** and tick
   **"Add Python to PATH"** during install.
3. **Internet for the first run only.** The AI model downloads once
   (a few hundred MB). After that it works with no internet.
4. The project folder (the `ED_updated` folder that has `embedding_experiment.py`
   and `triage_mixed_language_dataset.csv` inside it).

> You do NOT need to find or download any dataset. The file
> `triage_mixed_language_dataset.csv` is already inside the folder — that's our
> patient data. The script automatically uses 80% of it to teach the model and
> 20% to test how accurate it is.

---

## How to run it

### Step 1 — Open a terminal inside the project folder

**Windows:** open the `ED_updated` folder, click the address bar at the top,
type `powershell`, and press Enter. A terminal opens already inside the folder.

**Mac/Linux:** open Terminal and `cd` into the folder, e.g.
`cd ~/Downloads/ED_updated`.

### Step 2 — Install the libraries (one time)

First the main project libraries:

```
pip install -r requirements.txt
```

Then the one extra library this experiment needs:

```
pip install -r requirements-embedding.txt
```

This second one downloads the AI toolkit. It may take a few minutes. You only
do this once.

### Step 3 — Run the experiment

```
python embedding_experiment.py
```

The **first time**, it downloads the AI model (you'll see a progress bar). Wait
for it to finish. Every run after this works offline.

### Step 4 — Read the result

You'll see a table like this (numbers are examples — yours will differ):

```
RESULTS  (same split, same vitals, same classifier)
  A) Dictionary + BoW (current)   accuracy=88.80%   under-triage=5.0% (A+)  ...
  B) Embeddings only              accuracy=??.??%   under-triage=?.?% ...
  C) Both combined                accuracy=??.??%   under-triage=?.?% ...

Best method: ...
```

- **accuracy** = how often it got the triage level right. Higher is better.
- **under-triage** = how often it said a patient was *less* urgent than they
  really were. This is the dangerous mistake, so **lower is better** and it's
  the most important number.
- At the bottom it names the **best method** and gives a plain verdict.

It also saves the table to **`embedding_experiment_results.csv`** so you can
share it, and runs a **"synonym test"** showing whether Roman Urdu words like
`dard / drd / pain` actually land close together for the AI model.

---

## How to read the outcome

- If **B or C wins** (higher accuracy, lower under-triage) → the AI embedding
  idea helps. We should adopt it.
- If **A (Dictionary) wins** → the AI model didn't help on our Roman Urdu data.
  We keep the dictionary. That's still a useful result — we tested the idea
  properly and have numbers to prove it.
- The **synonym test** shows a "separation gap." A bigger gap means the AI
  understands our words well; a small gap means Roman Urdu is too unusual for
  this model (which is common, because these models mostly learned English and
  native-script Urdu, not Roman Urdu).

Whatever happens, screenshot the result table + the synonym test and send it
back — that's exactly what we need.

---

## If something goes wrong

**`python is not recognized`**
Python isn't installed or wasn't added to PATH. Reinstall from python.org and
tick "Add Python to PATH", then reopen the terminal.

**`ModuleNotFoundError: No module named 'sentence_transformers'`**
You skipped Step 2. Run `pip install -r requirements-embedding.txt`.

**`ModuleNotFoundError: No module named 'rapidfuzz'` (or pandas / sklearn)**
Run `pip install -r requirements.txt`.

**It's stuck downloading / "Could not load the model"**
The first run needs internet to fetch the AI model. Connect to WiFi and run it
again. Once it downloads, later runs don't need internet.

**`InconsistentVersionWarning` about scikit-learn**
Not an error, just a warning — you can ignore it for this experiment. (If you
want it gone: `pip install scikit-learn==1.6.1`.)

**Nothing prints / very slow**
Encoding ~1,200 patients on a slow laptop can take a minute or two the first
time. Give it a moment.

---

## Quick recap

| I want to...                                  | Type this                                     |
|-----------------------------------------------|-----------------------------------------------|
| Install main libraries (once)                 | `pip install -r requirements.txt`             |
| Install experiment library (once)             | `pip install -r requirements-embedding.txt`   |
| Run the experiment                            | `python embedding_experiment.py`              |
| Check Python is installed                     | `python --version`                            |
| See files in the folder                       | `dir` (Windows) / `ls` (Mac/Linux)            |

> Reminder: this is a student / research project, not a real medical device.
> It's for testing and learning, not for triaging real patients.
