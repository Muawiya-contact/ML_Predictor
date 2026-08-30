# Copilot instructions

Setup, run and troubleshooting instructions for this project live in
[`AGENTS.md`](../AGENTS.md) at the repository root. Read that file before
answering any question about installing or running this project.

Short version: create a venv, `pip install -r requirements.txt`, install
tkinter through the system package manager (it is not a pip package), start
Ollama and pull `llama3.2`, then `python triage_gui.py`. The first
prediction downloads a sentence encoder once; everything after that is
offline.
