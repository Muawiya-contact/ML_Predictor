"""Configurable directories; no application artifacts are modified."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(
    os.environ.get("TRIAGE_STUDY_OUTPUT", ROOT / "output/results/triage_study_run")
).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
MODEL_CACHE = Path(
    os.environ.get("TRIAGE_MODEL_CACHE", Path.home() / ".cache/huggingface/hub")
).resolve()
REFERENCES = Path(__file__).resolve().parent / "reference"
