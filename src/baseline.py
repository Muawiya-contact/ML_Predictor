"""
src/baseline.py
=======================================================================
Stratified 5-fold benchmark on the precomputed embeddings.
=======================================================================

Loads dataset.csv (185 rows) and complaint_embeddings.npy (185 x 384),
and cross-validates LogisticRegression, RandomForest and
HistGradientBoosting against both targets: Triage_Level (ordered, 4
classes) and Category (unordered, 11 classes).

WHY CROSS-VALIDATION RATHER THAN THE SUPPLIED SINGLE SPLIT
----------------------------------------------------------
The baseline this replaces reports 97.3% triage accuracy on a 37-row test
set. Four of those rows are Level 4, so a single flipped prediction moves
accuracy by 2.7 points and the per-class F1 for that class by far more.
A number like that is not wrong, it is just unfalsifiable from one split.
Five folds over all 185 rows, with the standard deviation printed beside
every mean, shows how much of the score is signal.

The department target is worse and the output says so per class:
Metabolic_Cardiovascular has 2 rows in the ENTIRE dataset, so whichever
fold holds its single test row decides its F1 outright.

    python -m src.baseline
    python -m src.baseline --embeddings professor_baseline/.../complaint_embeddings_direct_roman_urdu.npy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.encoders import StaticEncoder
from src.models import build_models, evaluate, safety_grade

warnings.filterwarnings("ignore")

BASE = os.path.join(_ROOT, "professor_baseline",
                    "How Effective is Embeddings Generator")
DEFAULT_DATA = os.path.join(BASE, "dataset.csv")
# The DIRECT Roman Urdu vectors, not complaint_embeddings.npy.
#
# Both files are 185x384 and neither is labelled, so the wrong one is easy
# to pick - and picking it is silent. complaint_embeddings.npy holds the
# TRANSLATED English text; the direct file holds the Roman Urdu. Measured
# by re-encoding the same 8 complaints with multilingual-e5-small and the
# "passage: " prefix: cosine 1.0000 against the direct file, 0.8356
# against the other. Since predict.py is fed raw Roman Urdu, training on
# the translated vectors would serve the classifier a space it was never
# fitted on - which is exactly what produced "chest pain -> Trauma at 17%"
# before this was caught. Pass --embeddings to benchmark the other arm.
DEFAULT_EMB = os.path.join(BASE, "complaint_embeddings_direct_roman_urdu.npy")

TARGETS = [
    # (column, human name, ordered?)
    ("Triage_Level", "TRIAGE LEVEL", True),
    ("Category", "DEPARTMENT (Category)", False),
]


def run_target(X, y, column, title, ordered, n_splits, seed):
    print("\n" + "=" * 78)
    print(f"{title}   -   {len(np.unique(y))} classes, {len(y)} rows")
    print("=" * 78)

    counts = pd.Series(y).value_counts().sort_index()
    print("  class support:", counts.to_dict())
    rare = counts[counts < n_splits]
    if len(rare):
        # StratifiedKFold cannot place a class with fewer rows than folds
        # into every fold; sklearn warns and carries on, which is easy to
        # miss in the output. Say it plainly instead.
        print(f"  [warn] {len(rare)} class(es) have fewer rows than folds "
              f"({n_splits}): {rare.to_dict()}")
        print("         Their per-class scores below are decided by one or "
              "two rows and should not be read as performance.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {}

    for name, _ in build_models(seed).items():
        accs, f1s, unders, overs = [], [], [], []
        oof_true, oof_pred = [], []

        for tr, te in skf.split(X, y):
            model = build_models(seed)[name]        # fresh per fold
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            m = evaluate(y[te], pred, ordered)
            accs.append(m["accuracy"])
            f1s.append(m["macro_f1"])
            if ordered:
                unders.append(m["under_triage_pct"])
                overs.append(m["over_triage_pct"])
            oof_true.extend(list(y[te]))
            oof_pred.extend(list(pred))

        row = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
        }
        if ordered:
            row["under_triage_mean"] = float(np.mean(unders))
            row["under_triage_std"] = float(np.std(unders))
            row["over_triage_mean"] = float(np.mean(overs))
            row["safety_grade"] = safety_grade(row["under_triage_mean"])
        # Out-of-fold predictions: every row predicted exactly once, by a
        # model that never saw it. This is the honest per-class table.
        row["_oof"] = evaluate(np.array(oof_true), np.array(oof_pred), ordered)
        results[name] = row

    hdr = f"  {'model':24s}{'accuracy':>18s}{'macro-F1':>18s}"
    if ordered:
        hdr += f"{'under-triage':>16s}{'grade':>7s}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, r in results.items():
        line = (f"  {name:24s}"
                f"{r['accuracy_mean']*100:11.1f}% ±{r['accuracy_std']*100:4.1f}"
                f"{r['macro_f1_mean']:13.3f} ±{r['macro_f1_std']:4.3f}")
        if ordered:
            line += (f"{r['under_triage_mean']:11.1f}% ±{r['under_triage_std']:3.1f}"
                     f"{r['safety_grade']:>7s}")
        print(line)

    best = max(results, key=lambda k: results[k]["macro_f1_mean"])
    print(f"\n  best by macro-F1: {best}")
    print(f"\n  per-class, out-of-fold ({best}):")
    rep = results[best]["_oof"]["report"]
    print(f"    {'class':26s}{'precision':>10s}{'recall':>9s}{'f1':>8s}{'support':>9s}")
    for k, v in rep.items():
        if isinstance(v, dict) and k not in ("macro avg", "weighted avg"):
            flag = "  <-- 1-2 rows" if v["support"] <= 2 else ""
            print(f"    {str(k):26s}{v['precision']:10.3f}{v['recall']:9.3f}"
                  f"{v['f1-score']:8.3f}{int(v['support']):9d}{flag}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--embeddings", default=DEFAULT_EMB)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    enc = StaticEncoder(args.embeddings, expected_rows=len(df))
    X = enc.encode(None)

    print("=" * 78)
    print("STATIC-EMBEDDING BASELINE  -  stratified 5-fold cross-validation")
    print("=" * 78)
    print(f"  dataset    : {os.path.basename(args.data)}  ({len(df)} rows)")
    print(f"  embeddings : {os.path.basename(args.embeddings)}  {X.shape}")
    print(f"  folds      : {args.folds}   seed: {args.seed}")

    out = {}
    for column, title, ordered in TARGETS:
        if column not in df.columns:
            print(f"\n[skip] no column {column!r} in {args.data}")
            continue
        res = run_target(X, df[column].values, column, title, ordered,
                         args.folds, args.seed)
        out[column] = {k: {kk: vv for kk, vv in v.items() if kk != "_oof"}
                       for k, v in res.items()}

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\n[ok] JSON written to {args.json_out}")


if __name__ == "__main__":
    main()
