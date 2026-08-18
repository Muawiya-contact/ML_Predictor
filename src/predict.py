"""
src/predict.py
=======================================================================
Live inference: raw Roman Urdu in, triage level and department out.
=======================================================================

Closes the dual pipeline. baseline.py evaluates on stored vectors;
this encodes typed text with the SAME model, prefix and normalization
those vectors were built with, then asks the persisted classifiers.

The encoder settings are checked against the training manifest before
anything is predicted. That check exists because the alternative has
already bitten this project once: a classifier served embeddings from a
different pipeline than it was fitted on does not crash, it just answers
worse, and the degradation is invisible without a controlled comparison.

    python -m src.predict "Mera sar dard ho raha hai"
    python -m src.predict "seena mein dard" "saans phool rahi hai"
    python -m src.predict --file complaints.txt
    python -m src.predict --interactive
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import joblib
import numpy as np

from src.encoders import EMBEDDING_DIM, EMBEDDING_MODEL, E5_PREFIX, DynamicEncoder
from src.train import MODEL_DIR

TRIAGE_LABELS = {
    1: "EMERGENCY   (immediate)",
    2: "URGENT      (within 15 min)",
    3: "STANDARD    (within 60 min)",
    4: "NON-URGENT  (can wait / redirect)",
}


class Predictor:
    """Loads the bundle once, then answers many complaints."""

    def __init__(self, model_dir: str = MODEL_DIR):
        mpath = os.path.join(model_dir, "manifest.json")
        if not os.path.exists(mpath):
            raise SystemExit(
                f"No manifest at {mpath}. Train first:\n"
                f"    .venv/bin/python -m src.train")
        with open(mpath, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        enc = self.manifest.get("encoder", {})
        # Refuse rather than degrade. A dimension or prefix mismatch means
        # the vectors this model was fitted on came from a different space.
        if enc.get("dim") != EMBEDDING_DIM or enc.get("prefix") != E5_PREFIX:
            raise SystemExit(
                f"Encoder mismatch. The bundle was trained with "
                f"{enc.get('model')} (dim {enc.get('dim')}, prefix "
                f"{enc.get('prefix')!r}) but src.encoders is configured for "
                f"{EMBEDDING_MODEL} (dim {EMBEDDING_DIM}, prefix "
                f"{E5_PREFIX!r}). Retrain, or restore the encoder settings.")

        self.models = {}
        for target, meta in self.manifest["targets"].items():
            path = os.path.join(model_dir, meta["file"])
            if not os.path.exists(path):
                raise SystemExit(f"Manifest lists {meta['file']} but it is missing.")
            self.models[target] = joblib.load(path)
        self.encoder = DynamicEncoder(enc.get("model", EMBEDDING_MODEL))

    def predict(self, texts):
        """Returns one dict per complaint."""
        texts = [str(t) for t in texts]
        X = self.encoder.encode(texts)
        out = [{"complaint": t} for t in texts]
        for target, model in self.models.items():
            pred = model.predict(X)
            proba = (model.predict_proba(X)
                     if hasattr(model, "predict_proba") else None)
            for i, row in enumerate(out):
                row[target] = pred[i]
                if proba is not None:
                    row[f"{target}_confidence"] = float(np.max(proba[i]))
        return out


def _render(row):
    print(f"\n  complaint : {row['complaint']}")
    lvl = row.get("Triage_Level")
    if lvl is not None:
        conf = row.get("Triage_Level_confidence")
        label = TRIAGE_LABELS.get(int(lvl), "")
        print(f"  triage    : Level {int(lvl)}  {label}"
              + (f"   [{conf*100:.1f}%]" if conf is not None else ""))
    dept = row.get("Category")
    if dept is not None:
        conf = row.get("Category_confidence")
        print(f"  department: {dept}"
              + (f"   [{conf*100:.1f}%]" if conf is not None else ""))
    low = [k for k in ("Triage_Level_confidence", "Category_confidence")
           if row.get(k) is not None and row[k] < 0.40]
    if low:
        print("  [warn] confidence under 40% - the model is guessing. This is a "
              "research prototype trained on 185 rows, not a medical device.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("complaints", nargs="*", help="raw Roman Urdu complaint(s)")
    p.add_argument("--file", help="text file, one complaint per line")
    p.add_argument("--model-dir", default=MODEL_DIR)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args()

    texts = list(args.complaints)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts += [ln.strip() for ln in f if ln.strip()]

    pred = Predictor(args.model_dir)

    if args.interactive:
        print("=" * 62)
        print("Roman Urdu complaint -> triage level + department")
        print(f"model dir: {args.model_dir}   |   type 'exit' to quit")
        print("=" * 62)
        while True:
            try:
                t = input("\nComplaint: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if t.lower() in ("exit", "quit"):
                break
            if not t:
                continue
            _render(pred.predict([t])[0])
        return

    if not texts:
        p.error("give at least one complaint, or --file, or --interactive")

    rows = pred.predict(texts)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        for row in rows:
            _render(row)


if __name__ == "__main__":
    main()
