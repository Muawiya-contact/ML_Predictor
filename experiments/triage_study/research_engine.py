"""Reproducible grouped model selection. Final test rows are used only in finalize().
Frozen text encoders never see labels. Imputation, scaling and PCA are fitted
inside each CV training fold. Cached matrices are keyed by fold membership.
"""

import os

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(k, "2")
from pathlib import Path
import argparse, hashlib, itertools, json, time, traceback
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from study_paths import OUTPUT as HERE

NUM = [
    "Age",
    "Heart_Rate",
    "Systolic_BP",
    "Diastolic_BP",
    "Temperature",
    "SpO2",
    "avpu_ord",
]
CAT = ["Gender", "Mode_of_Arrival", "ECG_Status"]
LABELS = [1, 2, 3]


class Features(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        view="structured",
        encoder=None,
        pca=64,
        text_weight=1.0,
        derived=False,
        whiten=False,
        solver="randomized",
        exclude_cat=(),
    ):
        self.view = view
        self.encoder = encoder
        self.pca = pca
        self.text_weight = text_weight
        self.derived = derived
        self.whiten = whiten
        self.solver = solver
        self.exclude_cat = exclude_cat

    def prepare(self, X):
        X = X.copy()
        X["avpu_ord"] = (
            X.AVPU.astype(str)
            .str.lower()
            .map(
                {
                    "a": 0,
                    "alert": 0,
                    "v": 1,
                    "voice": 1,
                    "verbal": 1,
                    "p": 2,
                    "pain": 2,
                    "u": 3,
                    "unresponsive": 3,
                }
            )
        )
        for c in NUM:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        if self.derived:
            X["pulse_pressure"] = X.Systolic_BP - X.Diastolic_BP
            X["heart_rate_over_sbp"] = X.Heart_Rate / X.Systolic_BP.replace(0, np.nan)
            X["mean_bp_proxy"] = (X.Systolic_BP + 2 * X.Diastolic_BP) / 3
        for c in CAT:
            X[c] = X[c].where(X[c].notna(), np.nan).astype(object)
        return X

    def embedding(self, X):
        return np.asarray(
            np.load(HERE / f"emb_{self.encoder}.npy", mmap_mode="r")[
                X.row_id.to_numpy(dtype=int)
            ],
            dtype=np.float64,
        )

    def fit(self, X, y=None):
        if self.view != "text":
            nums = NUM + (
                ["pulse_pressure", "heart_rate_over_sbp", "mean_bp_proxy"]
                if self.derived
                else []
            )
            self.structured_ = ColumnTransformer(
                [
                    (
                        "num",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        nums,
                    ),
                    (
                        "cat",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                (
                                    "encoder",
                                    OneHotEncoder(
                                        handle_unknown="ignore", sparse_output=False
                                    ),
                                ),
                            ]
                        ),
                        [c for c in CAT if c not in self.exclude_cat],
                    ),
                ]
            )
            self.structured_.fit(self.prepare(X))
        self.pca_ = None
        if self.view != "structured" and self.pca:
            self.pca_ = PCA(
                n_components=self.pca,
                whiten=self.whiten,
                svd_solver=self.solver,
                random_state=42,
            ).fit(self.embedding(X))
        return self

    def transform(self, X):
        parts = []
        if self.view != "text":
            parts.append(self.structured_.transform(self.prepare(X)))
        if self.view != "structured":
            e = self.embedding(X)
            if self.pca_ is not None:
                e = self.pca_.transform(e)
            parts.append(e * self.text_weight)
        return np.ascontiguousarray(np.hstack(parts), dtype=np.float64)


def ident(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:14]


def metrics(y, pred):
    report = classification_report(
        y, pred, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "qwk": cohen_kappa_score(y, pred, labels=LABELS, weights="quadratic"),
        "mae": float(np.abs(y - pred).mean()),
        "under_triage_rate": float((pred > y).mean()),
        "over_triage_rate": float((pred < y).mean()),
    }


def model_for(c):
    p = dict(c["params"])
    balance = p.pop("balance", False)
    if c["classifier"] == "logreg":
        m = LogisticRegression(max_iter=2000, random_state=42, **p)
    elif c["classifier"] == "hgb":
        m = HistGradientBoostingClassifier(random_state=42, **p)
    elif c["classifier"] == "rf":
        m = RandomForestClassifier(n_jobs=2, random_state=42, **p)
    else:
        raise ValueError(c["classifier"])
    return m, balance


def fit_model(c, X, y):
    model, balance = model_for(c)
    kw = {"sample_weight": compute_sample_weight("balanced", y)} if balance else {}
    model.fit(X, y, **kw)
    return model


def matrices(df, tr, va, features):
    key = ident(
        {
            "features": features,
            "train": tr.tolist(),
            "validation": va.tolist(),
            "input": hashlib.sha256(
                (HERE / "trimmed_dataset.csv").read_bytes()
            ).hexdigest(),
        }
    )
    dest = HERE / "matrix_cache"
    dest.mkdir(exist_ok=True)
    path = dest / f"{key}.npz"
    if path.exists():
        z = np.load(path)
        return z["train"], z["validation"]
    transformer = Features(**features).fit(df.iloc[tr])
    a = transformer.transform(df.iloc[tr])
    b = transformer.transform(df.iloc[va])
    np.savez(path, train=a, validation=b)
    return a, b


def evaluate_cv(config, folds=3):
    tag = ident(config)
    dest = HERE / f"cv{folds}"
    dest.mkdir(exist_ok=True)
    resultfile = dest / f"{tag}.json"
    if resultfile.exists():
        return json.loads(resultfile.read_text())
    df = pd.read_csv(HERE / "dataset_with_splits.csv")
    dev = np.flatnonzero(df.partition.eq("development"))
    y = df.Labels.to_numpy(dtype=int)
    probabilities = np.full((len(df), 3), np.nan)
    foldrows = []
    start = time.monotonic()
    for fold in range(folds):
        va = np.flatnonzero(df[f"cv{folds}"].eq(fold))
        tr = np.setdiff1d(dev, va)
        assert not set(df.iloc[tr].group) & set(df.iloc[va].group)
        a, b = matrices(df, tr, va, config["features"])
        m = fit_model(config, a, y[tr])
        prob = m.predict_proba(b)
        assert list(m.classes_) == LABELS
        probabilities[va] = prob
        foldrows.append(metrics(y[va], np.argmax(prob, axis=1) + 1))
        print(
            f'{tag} {config["classifier"]} {config["features"]} fold {fold+1}/{folds}: F1={foldrows[-1]["macro_f1"]:.4f}',
            flush=True,
        )
    result = {
        "id": tag,
        "config": config,
        "folds": folds,
        "mean": {k: float(np.mean([r[k] for r in foldrows])) for k in foldrows[0]},
        "std": {k: float(np.std([r[k] for r in foldrows])) for k in foldrows[0]},
        "per_fold": foldrows,
        "seconds": time.monotonic() - start,
        "pooled_oof": metrics(y[dev], np.argmax(probabilities[dev], axis=1) + 1),
    }
    np.save(dest / f"{tag}_oof.npy", probabilities)
    resultfile.write_text(json.dumps(result, indent=2))
    return result


def config(clf, features, **params):
    return {"classifier": clf, "features": features, "params": params}


def candidates(stage):
    out = []
    if stage == "structured":
        for derived in (False, True):
            f = {"view": "structured", "derived": derived}
            for C, b in itertools.product((0.1, 1, 10), (False, True)):
                out.append(config("logreg", f, C=C, balance=b))
            for leaves, l2, b in [
                (7, 1, False),
                (15, 1, False),
                (31, 0, False),
                (15, 1, True),
                (31, 1, True),
                (15, 10, False),
            ]:
                out.append(
                    config(
                        "hgb",
                        f,
                        max_iter=300,
                        max_leaf_nodes=leaves,
                        l2_regularization=l2,
                        learning_rate=0.1,
                        balance=b,
                    )
                )
            for leaf, b in [(1, False), (3, False), (3, True)]:
                out.append(
                    config(
                        "rf",
                        f,
                        n_estimators=300,
                        min_samples_leaf=leaf,
                        max_features=0.7,
                        balance=b,
                    )
                )
    else:
        for encoder in ["sapbert_concept", "mpnet_concept", "minilm_complaint"]:
            if not (HERE / f"emb_{encoder}.npy").exists():
                continue
            for view in ("text", "fused"):
                for pc, C in itertools.product((32, 64, 128, 0), (0.1, 1, 10)):
                    out.append(
                        config(
                            "logreg",
                            {"view": view, "encoder": encoder, "pca": pc},
                            C=C,
                            balance=True,
                        )
                    )
            text_features = {"view": "text", "encoder": encoder, "pca": 64}
            out.append(
                config(
                    "hgb",
                    text_features,
                    max_iter=250,
                    max_leaf_nodes=15,
                    l2_regularization=1,
                    learning_rate=0.1,
                )
            )
            out.append(
                config(
                    "rf",
                    text_features,
                    n_estimators=300,
                    min_samples_leaf=3,
                    max_features=0.7,
                    balance=True,
                )
            )
            for pc, b in itertools.product((32, 64, 128), (False, True)):
                f = {"view": "fused", "encoder": encoder, "pca": pc, "derived": True}
                out.append(
                    config(
                        "hgb",
                        f,
                        max_iter=250,
                        max_leaf_nodes=15,
                        l2_regularization=1,
                        learning_rate=0.1,
                        balance=b,
                    )
                )
            for pc in (64, 128):
                out.append(
                    config(
                        "rf",
                        {"view": "fused", "encoder": encoder, "pca": pc},
                        n_estimators=300,
                        min_samples_leaf=3,
                        max_features=0.7,
                        balance=True,
                    )
                )
        # Exact uploaded configurations retained as baseline references (PCA solver defaults in source).
        f = {"view": "fused", "encoder": "sapbert_concept", "pca": 64}
        out.extend(
            [
                config("hgb", f, max_iter=300, learning_rate=0.1),
                config("logreg", f, C=1, balance=True),
            ]
        )
    return list({ident(c): c for c in out}.values())


def export_board(folds):
    results = [json.loads(p.read_text()) for p in (HERE / f"cv{folds}").glob("*.json")]
    if results:
        pd.DataFrame(
            [
                {
                    "id": r["id"],
                    "classifier": r["config"]["classifier"],
                    **r["config"]["features"],
                    **r["mean"],
                    "macro_f1_std": r["std"]["macro_f1"],
                    "seconds": r["seconds"],
                    "parameters": json.dumps(r["config"]["params"]),
                }
                for r in results
            ]
        ).sort_values("macro_f1", ascending=False).to_csv(
            HERE / f"cv{folds}_leaderboard.csv", index=False
        )
    return results


def screen(stage):
    configs = candidates(stage)
    (HERE / f"{stage}_search_plan.json").write_text(json.dumps(configs, indent=2))
    for i, c in enumerate(configs):
        print(f"SCREEN {stage} {i+1}/{len(configs)}", flush=True)
        try:
            evaluate_cv(c, 3)
        except Exception:
            with (HERE / "failures.log").open("a") as f:
                f.write(json.dumps(c) + "\n" + traceback.format_exc() + "\n")
            print(traceback.format_exc(), flush=True)
        export_board(3)


def refine():
    # Complete any comparison families added while an earlier screen was running.
    for c in candidates("text"):
        evaluate_cv(c, 3)
    results = export_board(3)
    ranked = sorted(results, key=lambda r: r["mean"]["macro_f1"], reverse=True)
    shortlist = ranked[:8]
    for clf in ["logreg", "hgb", "rf"]:
        for view in ["structured", "text", "fused"]:
            eligible = [
                r
                for r in ranked
                if r["config"]["classifier"] == clf
                and r["config"]["features"]["view"] == view
            ]
            if eligible:
                shortlist.append(eligible[0])
    shortlist = list({r["id"]: r for r in shortlist}.values())
    (HERE / "shortlist.json").write_text(
        json.dumps([r["config"] for r in shortlist], indent=2)
    )
    for r in shortlist:
        evaluate_cv(r["config"], 5)
        export_board(5)
    # Tune regularization / relative embedding weight for the best fused logistic model.
    fused = [
        r
        for r in ranked
        if r["config"]["classifier"] == "logreg"
        and r["config"]["features"]["view"] == "fused"
    ]
    if fused:
        best = fused[0]["config"]
        for weight, balance in itertools.product((0.25, 1, 4), (False, True)):
            c = json.loads(json.dumps(best))
            c["features"]["text_weight"] = weight
            c["params"]["balance"] = balance
            evaluate_cv(c, 5)
            export_board(5)
    print("REFINEMENT COMPLETE", flush=True)


def finalize():
    """Freeze a selection using OOF predictions before reading test outcomes."""
    if (HERE / "final_results.json").exists():
        print("Final test already evaluated; refusing repeated selection.")
        return
    results = export_board(5)
    ranked = sorted(results, key=lambda r: r["mean"]["macro_f1"], reverse=True)
    df = pd.read_csv(HERE / "dataset_with_splits.csv")
    dev = np.flatnonzero(df.partition.eq("development"))
    test = np.flatnonzero(df.partition.eq("test"))
    y = df.Labels.to_numpy(dtype=int)
    # Evaluate single models, an equal vote across family winners, and modest class offsets on OOF only.
    family = []
    for clf in ["logreg", "hgb", "rf"]:
        eligible = [r for r in ranked if r["config"]["classifier"] == clf]
        if eligible:
            family.append(eligible[0])
    schemes = [([r], [1.0]) for r in ranked[:5]]
    if len(family) > 1:
        schemes.append((family, [1 / len(family)] * len(family)))
        for i in range(len(family)):
            weights = np.ones(len(family))
            weights[i] = 2
            weights /= weights.sum()
            schemes.append((family, weights.tolist()))
    choices = []
    for members, weights in schemes:
        prob = sum(
            w * np.load(HERE / "cv5" / f'{r["id"]}_oof.npy')[dev]
            for r, w in zip(members, weights)
        )
        for factors in itertools.product((0.8, 1.0, 1.2), repeat=2):
            multiplier = np.array([factors[0], 1.0, factors[1]])
            pred = np.argmax(prob * multiplier, axis=1) + 1
            scores = metrics(y[dev], pred)
            choices.append(
                {
                    "members": [r["id"] for r in members],
                    "weights": weights,
                    "class_probability_multipliers": multiplier.tolist(),
                    "oof_metrics": scores,
                }
            )
    selection = max(
        choices,
        key=lambda s: (s["oof_metrics"]["macro_f1"], s["oof_metrics"]["accuracy"]),
    )
    (HERE / "frozen_selection.json").write_text(json.dumps(selection, indent=2))
    by_id = {r["id"]: r for r in results}
    testprobs = {}
    models = HERE / "models"
    models.mkdir(exist_ok=True)
    # Also report each family winner. This set was fixed using development CV only.
    reportids = list(
        dict.fromkeys(
            selection["members"] + [r["id"] for r in family] + [ranked[0]["id"]]
        )
    )
    allconfigs = {i: by_id[i]["config"] for i in reportids}
    # Source baselines: balanced LR and original HGB on all three feature views.
    baselineids = []
    for view in ["structured", "text", "fused"]:
        f = {"view": view, "encoder": "sapbert_concept", "pca": 64, "solver": "auto"}
        for clf, params in [
            ("logreg", {"C": 1, "balance": True}),
            ("hgb", {"max_iter": 300, "learning_rate": 0.1}),
        ]:
            c = config(clf, f, **params)
            i = ident(c)
            allconfigs[i] = c
            baselineids.append(i)
    final = []
    for i, c in allconfigs.items():
        print("FINAL FIT", i, c, flush=True)
        transformer = Features(**c["features"]).fit(df.iloc[dev])
        a = transformer.transform(df.iloc[dev])
        b = transformer.transform(df.iloc[test])
        m = fit_model(c, a, y[dev])
        prob = m.predict_proba(b)
        testprobs[i] = prob
        pred = np.argmax(prob, axis=1) + 1
        joblib.dump(
            {"features": transformer, "classifier": m, "config": c, "labels": LABELS},
            models / f"{i}.joblib",
        )
        pd.DataFrame(
            {
                "row_id": test,
                "true": y[test],
                "predicted": pred,
                **{f"p_{k}": prob[:, k - 1] for k in LABELS},
            }
        ).to_csv(HERE / f"final_{i}_predictions.csv", index=False)
        final.append(
            {
                "id": i,
                "role": "source_baseline" if i in baselineids else "cv_family_winner",
                "config": c,
                "feature_count": int(a.shape[1]),
                "metrics": metrics(y[test], pred),
                "report": classification_report(
                    y[test], pred, labels=LABELS, output_dict=True, zero_division=0
                ),
                "confusion": confusion_matrix(y[test], pred, labels=LABELS).tolist(),
            }
        )
    prob = sum(
        w * testprobs[i] for i, w in zip(selection["members"], selection["weights"])
    )
    pred = (
        np.argmax(prob * np.array(selection["class_probability_multipliers"]), axis=1)
        + 1
    )
    selected = {
        "id": "selected_system",
        "role": "selected_on_development_only",
        "selection": selection,
        "metrics": metrics(y[test], pred),
        "report": classification_report(
            y[test], pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion": confusion_matrix(y[test], pred, labels=LABELS).tolist(),
    }
    final.append(selected)
    pd.DataFrame(
        {
            "row_id": test,
            "true": y[test],
            "predicted": pred,
            **{f"p_{k}": prob[:, k - 1] for k in LABELS},
        }
    ).to_csv(HERE / "final_selected_system_predictions.csv", index=False)
    # Group bootstrap preserves within-concept dependence; no model selection uses these intervals.
    groups = df.iloc[test].group.to_numpy()
    unique = np.unique(groups)
    groupindices = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(2026)
    boot = []
    for _ in range(1000):
        ix = np.concatenate(
            [groupindices[j] for j in rng.integers(0, len(unique), len(unique))]
        )
        boot.append(
            [
                accuracy_score(y[test][ix], pred[ix]),
                f1_score(
                    y[test][ix],
                    pred[ix],
                    average="macro",
                    labels=LABELS,
                    zero_division=0,
                ),
            ]
        )
    selected["group_bootstrap_95_ci"] = {
        k: np.percentile(np.array(boot)[:, j], [2.5, 97.5]).tolist()
        for j, k in enumerate(["accuracy", "macro_f1"])
    }
    (HERE / "final_results.json").write_text(json.dumps(final, indent=2))
    pd.DataFrame(
        [{"id": r["id"], "role": r["role"], **r["metrics"]} for r in final]
    ).to_csv(HERE / "final_metrics.csv", index=False)
    print("SELECTED SYSTEM", json.dumps(selected["metrics"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["structured", "text", "refine", "finalize"])
    a = p.parse_args()
    import research_engine as engine

    if a.stage in ["structured", "text"]:
        engine.screen(a.stage)
    elif a.stage == "refine":
        engine.refine()
    else:
        engine.finalize()
