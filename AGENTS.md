# AGENTS.md — setting this project up on a new machine

You are an AI coding assistant. Someone has just been handed this project and
wants to run it on their own laptop. This file tells you how. Follow it in
order; ask the user to run the commands, and read the output before moving on.

**One sentence about the project:** a patient's complaint is typed in Roman
Urdu, translated to English by a language model running on the user's own
machine, checked, and scored into an emergency triage level 1–4. Nothing is
sent over the internet at prediction time.

---

## Before anything else: check what is already there

Run this and read the answers rather than assuming:

```bash
python --version                 # need 3.10 or newer
python -c "import tkinter"       # must not error - see step 2
ollama --version                 # the local translator
ollama list                      # which models are already pulled
```

Do not skip the tkinter line. It is the step most likely to fail, and it
fails differently on every operating system.

---

## Step 1 — Python packages

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

This pulls in PyTorch as a dependency of sentence-transformers and takes a
few minutes. The CPU build is all that is needed — do not install a CUDA
build.

**scikit-learn is pinned on purpose.** The trained models ship as pickle
files. A different scikit-learn version can load them with a warning and
give different answers. If the user wants a different version, they must
retrain (`python train_embedding_pipeline.py`), not ignore the warning.

---

## Step 2 — tkinter (the desktop window)

The GUI uses tkinter. It is part of Python's standard library but several
Linux distributions ship it separately.

| System | What to do |
|---|---|
| Windows / macOS (python.org installer) | Already included. Nothing to do. |
| Ubuntu / Debian | `sudo apt install python3-tk` |
| Fedora / RHEL | `sudo dnf install python3-tkinter` |
| macOS (Homebrew python) | `brew install python-tk` |

Verify with `python -c "import tkinter"` — silence means success.

> **`run_gui.sh` is Linux-specific and probably will not help here.** It
> exists to point Python at a private copy of the Tk runtime kept inside
> `.venv/` on the original developer's Fedora machine. That copy is not in
> the repository. On any other machine, install tkinter as above and run
> `python triage_gui.py` directly.

---

## Step 3 — Ollama, the local translator

The app translates every complaint before scoring it, using a model running
on this machine. Without it, the app will not produce a prediction — by
design, it reports the problem instead of guessing.

1. If `ollama --version` failed, install it from <https://ollama.com/download>.
2. Start the service and leave it running:
   ```bash
   ollama serve
   ```
   On Linux, `sudo systemctl enable --now ollama` makes it survive a reboot.
3. Pull the model (about 2 GB, one time, needs internet):
   ```bash
   ollama pull llama3.2
   ```

Any of these also work if one is already installed: `qwen2.5`, `mistral`,
`gemma2`, `phi3`. The app picks whichever it finds and prints which one it
used — it does not force a download when a usable model is already present.

---

## Step 4 — First run downloads the sentence encoder

The first prediction downloads a small multilingual encoder (a few hundred
MB) into the Hugging Face cache and reuses it forever after. **This one step
needs internet.** Everything after it is offline.

Warm it up before demoing, so a live audience does not sit through a
download:

```bash
python run_inference.py --check
python run_inference.py "seena mein dard hai"
```

---

## Step 5 — Run it

```bash
python triage_gui.py
```

A window opens with six tabs. Type a complaint into **Triage a Patient** and
press the button. A single prediction takes roughly 11 seconds, almost all
of it translation.

---

## Verifying the install actually works

Two test suites ship with the project. Both need Ollama running.

```bash
python tests/audit_pipeline.py     # 16 checks - translation, safety gate, dictionary
python tests/audit_gui.py          # 17 checks - builds the real window, drives every tab
```

All checks passing is the definition of "correctly installed". If any fail,
read the failure line — each one states what it expected and what it got.

---

## When something goes wrong

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: tkinter` | Step 2. It is not a pip package. |
| "Ollama is not reachable" | The service is not running. `ollama serve`. |
| Translation times out on the very first try | Cold start under load. Run it again; the second attempt is usually instant. Do not start a batch at the same moment as the first prediction — both load models onto the same CPU. |
| `InconsistentVersionWarning` on startup | scikit-learn differs from the pinned version. Either match the pin or retrain. Do not ignore it: the models are pickles. |
| Batch of 100 patients "hangs" | It is not hung. One translation per row at ~11 seconds means about 18 minutes. The progress bar shows the row count and an estimate. |
| "Anatomical check failed" | Working as intended. The translation moved the complaint to a different body part, so no prediction was made. |

---

## Things not to do

- **Do not retrain to "fix" a problem.** The trained models are committed to
  the repository and are the ones that were evaluated. Retraining replaces
  them with something unmeasured.
- **Do not ZIP the `.venv/` folder** when sharing this project. It is 1.6 GB
  of compiled Linux binaries and will not work on another machine. Send the
  repository without it; step 1 rebuilds it in a few minutes.
- **Do not add an API key or a cloud model.** Every stage runs locally on
  purpose. There is no key to set and nothing to sign up for.
- **Do not change the confidence cap or the anatomical check** to make
  outputs look better. Both exist to stop the app looking confident when it
  should not.

---

## Facts about this project a helper should know

- **The training data is synthetic**, generated by a script. It is not real
  patient data and must not be described as such.
- **This is a research prototype, not a medical device.** It has not been
  clinically validated and no output should be acted on without a clinician.
- **The safety check is deterministic**, not a similarity score. If the
  complaint names a body part, the English translation must contain that
  body part or the prediction is refused. An earlier version compared the
  two sentences numerically and was removed after measurement showed a
  correct translation (0.8054) and a completely wrong one (0.7922) scoring
  almost the same.
- **`docs/ML_Predictor_Manual.pdf`** is a four-page operator manual covering
  the workflow, how to read the output, and the system's limits. Read it
  before answering questions about behaviour.
