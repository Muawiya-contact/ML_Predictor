"""
src/models.py
=======================================================================
The three classifiers, and the metrics this project actually cares about.
=======================================================================

WHY THESE THREE, AND NO XGBOOST OR LIGHTGBM
-------------------------------------------
HistGradientBoostingClassifier is gradient boosting from scikit-learn, so
the benchmark gets a boosted-tree entry with no extra dependency and no
install step on a machine that has to run offline. On 185 rows the
difference between it and XGBoost is far smaller than the fold-to-fold
noise, which the cross-validation standard deviations in baseline.py make
visible.

WHY UNDER-TRIAGE IS REPORTED SEPARATELY
---------------------------------------
Accuracy treats every mistake alike. Triage does not. Calling a Level 1
patient Level 3 (under-triage) can kill; calling a Level 3 patient
Level 1 (over-triage) wastes a bed. The two are reported separately, and
only for the ordered triage target - for department routing the classes
are unordered, so "under" and "over" are meaningless there and are
omitted rather than computed and quietly misread.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

RANDOM_STATE = 42


def build_models(random_state: int = RANDOM_STATE) -> Dict[str, object]:
    """The benchmark line-up. Fresh instances every call - reusing a fitted
    estimator across folds would leak the previous fold's fit."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            # 185 rows over 11 departments leaves classes with 2 examples.
            # Without balancing, the model can score respectably by never
            # predicting them at all.
            class_weight="balanced",
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.1,
            # early_stopping is OFF deliberately. With it on, the estimator
            # carves its own stratified validation slice out of the training
            # fold, and the department target has a class with 2 rows in the
            # whole dataset - once cross-validation puts one of them in test,
            # the inner split sees a single member and sklearn raises
            # "The least populated classes in y have only 1 member".
            # On 185 rows early stopping buys little anyway; max_iter is the
            # regularizer that matters here.
            early_stopping=False,
            random_state=random_state,
        ),
    }


def triage_error_rates(y_true, y_pred) -> Dict[str, float]:
    """Under- and over-triage percentages for the ORDERED triage target.

    Levels run 1 (most urgent) to 4 (least). Predicting a HIGHER number
    than the truth means the patient was called less urgent than they
    are - that is under-triage, and it is the dangerous direction.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"under_triage_pct": 0.0, "over_triage_pct": 0.0}
    return {
        "under_triage_pct": 100.0 * float(np.mean(y_pred > y_true)),
        "over_triage_pct": 100.0 * float(np.mean(y_pred < y_true)),
    }


def safety_grade(under_pct: float) -> str:
    """Same bands the deployed triage system already reports against."""
    if under_pct < 5:
        return "A+"
    if under_pct < 10:
        return "A"
    if under_pct < 15:
        return "B"
    if under_pct < 20:
        return "C"
    return "F"


def evaluate(y_true, y_pred, ordered: bool) -> Dict[str, object]:
    """Accuracy, macro-F1, per-class table, and triage errors when ordered."""
    out: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "report": classification_report(y_true, y_pred, zero_division=0,
                                        output_dict=True),
    }
    if ordered:
        rates = triage_error_rates(y_true, y_pred)
        out.update(rates)
        out["safety_grade"] = safety_grade(rates["under_triage_pct"])
    return out
