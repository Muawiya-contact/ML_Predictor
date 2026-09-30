"""Resume model comparison after preparation and offline embedding generation."""

import subprocess
import sys
from pathlib import Path
from study_paths import OUTPUT

SCRIPTS = Path(__file__).resolve().parent
required = ["dataset_with_splits.csv"] + [
    f"emb_{name}.json"
    for name in ("sapbert_concept", "mpnet_concept", "minilm_complaint")
]
missing = [name for name in required if not (OUTPUT / name).exists()]
if missing:
    raise SystemExit(
        "Prepare data and generate embeddings first. Missing: " + ", ".join(missing)
    )
# POSIX lock prevents concurrent writers from racing on trial/result files.
import fcntl

with (OUTPUT / "study.lock").open("w") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another study worker is running.")
    steps = [
        ("validate_protocol.py",),
        ("research_engine.py", "structured"),
        ("research_engine.py", "text"),
        ("research_engine.py", "refine"),
        ("sensitivity.py",),
        ("diagnostics.py",),
    ]
    if not (OUTPUT / "final_results.json").exists():
        steps.append(("research_engine.py", "finalize"))
    for script, *args in steps:
        subprocess.run([sys.executable, str(SCRIPTS / script), *args], check=True)
    print(
        "Study complete. See final_metrics.csv; report reproduction is documented in README.md."
    )
