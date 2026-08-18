"""
src/train.py
=======================================================================
Fit the production classifiers on all 185 rows and persist them.
=======================================================================

baseline.py answers "how well does this generalise" with cross-validation.
This answers "what do we ship" - the same estimator refitted on every row,
because holding 20% back permanently would throw away a fifth of a very
small dataset for no benefit once the model choice is settled.

The bundle records which encoder settings produced the training vectors.
predict.py refuses to run against a mismatch rather than silently feeding
the classifier a different embedding space - the failure mode that made
the English triage model look broken earlier in this project.

    python -m src.train
    python -m src.train --model RandomForest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import joblib
import numpy as np
import pandas as pd

from src.baseline import DEFAULT_DATA, DEFAULT_EMB, TARGETS
from src.encoders import EMBEDDING_DIM, EMBEDDING_MODEL, E5_PREFIX, StaticEncoder
from src.models import build_models

MODEL_DIR = os.path.join(_ROOT, "models_src")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--embeddings", default=DEFAULT_EMB)
    p.add_argument("--model", default="RandomForest",
                   choices=list(build_models().keys()),
                   help="estimator to ship (baseline.py picks the winner)")
    p.add_argument("--out-dir", default=MODEL_DIR)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    X = StaticEncoder(args.embeddings, expected_rows=len(df)).encode(None)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[ok] {len(df)} rows, embeddings {X.shape}, model {args.model}")
    manifest = {
        "trained_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "dataset": os.path.basename(args.data),
        "rows": int(len(df)),
        "estimator": args.model,
        # predict.py checks these before trusting a live embedding.
        "encoder": {
            "model": EMBEDDING_MODEL,
            "dim": EMBEDDING_DIM,
            "prefix": E5_PREFIX,
            "normalize": True,
        },
        "targets": {},
    }

    for column, title, _ordered in TARGETS:
        if column not in df.columns:
            continue
        y = df[column].values
        est = build_models()[args.model]
        est.fit(X, y)
        path = os.path.join(args.out_dir, f"{column.lower()}_model.pkl")
        joblib.dump(est, path)
        manifest["targets"][column] = {
            "classes": [str(c) for c in est.classes_],
            "file": os.path.basename(path),
            "train_accuracy": float(est.score(X, y)),
        }
        print(f"  {title:24s} -> {os.path.basename(path)}  "
              f"({len(est.classes_)} classes, train acc "
              f"{est.score(X, y)*100:.1f}%)")

    mpath = os.path.join(args.out_dir, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ok] manifest -> {mpath}")
    print("     NOTE train accuracy is fitted-on-itself and will read high; "
          "the honest numbers are the cross-validated ones from src.baseline.")


if __name__ == "__main__":
    main()
