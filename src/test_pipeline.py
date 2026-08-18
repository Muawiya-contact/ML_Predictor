"""
src/test_pipeline.py
=======================================================================
Validation for the dual pipeline. Run it after any change to encoders.
=======================================================================

Checks the invariants that make the two pipelines comparable, rather than
re-checking accuracy (baseline.py owns that). The important one is the
last: static and dynamic encoders must place the same complaint in the
same place, or a model validated offline is not the model answering live.

    .venv/bin/python -m src.test_pipeline
    .venv/bin/python -m src.test_pipeline --skip-dynamic   # no network
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from src.baseline import DEFAULT_DATA, DEFAULT_EMB
from src.encoders import EMBEDDING_DIM, StaticEncoder
from src.models import build_models, safety_grade, triage_error_rates

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, condition, detail=""):
    _results.append((PASS if condition else FAIL, name, detail))
    print(f"  [{PASS if condition else FAIL}] {name}" + (f"  {detail}" if detail else ""))
    return condition


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-dynamic", action="store_true",
                   help="skip the live-encoder checks (they need the model)")
    args = p.parse_args()

    print("=" * 70)
    print("DUAL-PIPELINE VALIDATION")
    print("=" * 70)

    df = pd.read_csv(DEFAULT_DATA)
    print("\nStatic path")
    enc = StaticEncoder(DEFAULT_EMB, expected_rows=len(df))
    X = enc.encode(None)
    check("embeddings load", X is not None, str(X.shape))
    check("row count matches dataset", len(X) == len(df),
          f"{len(X)} vs {len(df)}")
    check("dimension is 384", X.shape[1] == EMBEDDING_DIM)
    check("dtype float32", X.dtype == np.float32, str(X.dtype))
    check("no NaN or inf", bool(np.isfinite(X).all()))
    norms = np.linalg.norm(X, axis=1)
    check("vectors L2-normalised", bool(np.allclose(norms, 1.0, atol=1e-3)),
          f"mean norm {norms.mean():.4f}")

    print("\nEmpty-file guard")
    empty = os.path.join(os.path.dirname(DEFAULT_EMB), "roman_urdu_embeddings.npy")
    if os.path.exists(empty):
        try:
            StaticEncoder(empty)
            check("empty .npy rejected", False, "it loaded silently")
        except ValueError as e:
            check("empty .npy rejected", True, str(e)[:52] + "...")
    else:
        print("  [skip] no empty file present to test against")

    print("\nRow-count mismatch guard")
    try:
        StaticEncoder(DEFAULT_EMB, expected_rows=len(df) + 1)
        check("mismatched row count rejected", False, "accepted silently")
    except ValueError:
        check("mismatched row count rejected", True)

    print("\nMetrics")
    r = triage_error_rates([1, 1, 2, 3], [1, 2, 2, 3])   # one under-triage of 4
    check("under-triage direction correct", abs(r["under_triage_pct"] - 25.0) < 1e-6,
          f"{r['under_triage_pct']:.1f}%")
    r2 = triage_error_rates([3, 3], [1, 1])              # both over-triaged
    check("over-triage direction correct", abs(r2["over_triage_pct"] - 100.0) < 1e-6,
          f"{r2['over_triage_pct']:.1f}%")
    check("safety grades band correctly",
          safety_grade(4) == "A+" and safety_grade(9) == "A" and safety_grade(25) == "F")

    print("\nModels")
    models = build_models()
    check("three estimators built", len(models) == 3, ", ".join(models))
    for name, est in models.items():
        est.fit(X[:120], df["Triage_Level"].values[:120])
        pred = est.predict(X[120:])
        check(f"{name} predicts correct shape", pred.shape == (len(X) - 120,))

    if not args.skip_dynamic:
        print("\nDynamic path  (needs multilingual-e5-small)")
        try:
            from src.encoders import DynamicEncoder
            dyn = DynamicEncoder()
            texts = df["Roman Urdu Complaint"].astype(str).tolist()[:8]
            D = dyn.encode(texts)
            check("dynamic encode shape", D.shape == (8, EMBEDDING_DIM), str(D.shape))
            check("dynamic dtype float32", D.dtype == np.float32)
            check("dynamic L2-normalised",
                  bool(np.allclose(np.linalg.norm(D, axis=1), 1.0, atol=1e-3)))
            check("empty string handled", dyn.encode([""]).shape == (1, EMBEDDING_DIM))

            # THE check this file exists for: re-encoding a stored complaint
            # should land on its stored vector. If this drifts, offline
            # evaluation stops predicting live behaviour.
            sims = [float(np.dot(D[i], X[i])) for i in range(8)]
            mean_sim = float(np.mean(sims))
            check("static and dynamic agree (cos > 0.98)", mean_sim > 0.98,
                  f"mean cosine {mean_sim:.4f}")
        except Exception as e:
            check("dynamic encoder available", False, f"{type(e).__name__}: {e}")
    else:
        print("\n  [skip] dynamic checks (--skip-dynamic)")

    n_fail = sum(1 for s, _, _ in _results if s == FAIL)
    print("\n" + "=" * 70)
    print(f"{len(_results) - n_fail}/{len(_results)} passed"
          + (f", {n_fail} FAILED" if n_fail else " - all good"))
    print("=" * 70)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
