# Run the current four-level application

Follow [README.md](README.md#run-locally) for Python/tkinter, pinned packages,
CPU PyTorch and the one-time downloads of local Ollama/Qwen and SapBERT.

```bash
source .venv/bin/activate
python run_inference.py --check
python triage_gui.py
```

On the original Linux installation, `./run_gui.sh` sets the private Tk runtime
paths. `./run_local.sh` also checks Python, Ollama and the serving bundle.

The active bundle is `triage_model_sapbert/`; levels are 0 Emergency, 1 Urgent,
2 Standard, 3 Non-urgent. The GUI, interactive CLI and batch CLI use it by default.
Its classifier and saved results are identified in the bundle manifest.

```bash
python run_inference.py 'seena mein dard hai' --age 65 --heart-rate 118
python predict_batch.py patients.csv results.xlsx
```

See [the detailed GUI guide](docs/SapBERT_GUI.md) for inputs, explanations,
local encoder paths and error handling. Cold local translation can take several
minutes; it runs in a worker so the GUI remains responsive. No usable complaint
produces a 50% placeholder with no level. Translation/anatomical failures withhold
predictions. Missing measurements use fitted statistics with a note.

An encoder cache error requires the exact SapBERT snapshot, not another model
with a similar dimension. `SAPBERT_MODEL_PATH` points to that local snapshot.
A classifier/manifest mismatch requires restoring the matching evaluated bundle;
retraining is not an installation repair. See [the research protocol](experiments/triage_study/FOUR_LEVEL.md)
when intentionally running a new experiment.

Earlier operator PDFs and installation descriptions in Git history refer to
older MiniLM and 1-based labels. Use this guide for the current application.
