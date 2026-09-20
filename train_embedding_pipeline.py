"""Compare raw and preprocessed sentence embeddings with structured features.

Both configurations use Logistic Regression. B embeds raw text; C removes
learned stop words and, for Roman Urdu, applies dictionary normalization.
Stop words for evaluation are learned from training rows only. The default
input is the English synthetic subset used by the GUI. Training is an
explicit research operation; it is not required to run the saved model.
"""

import argparse
import json
import os
import sys
import warnings

# See prediction.py: keeps `import triage_pipeline` working regardless of
# how this script was launched.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from stopwords import learn_stopwords, save_stopwords, summarize
from triage_pipeline import (
    EMBEDDING_MODEL_NAME,
    NUMERICAL_FEATURES,
    encode_categoricals,
    make_console_safe,
    normalize_roman_urdu,
    preprocess_corpus_for_embedding,
    project_path,
    resolve_project_file,
)

warnings.filterwarnings("ignore")
make_console_safe()

DATA_FILE = "cardiac_english_2252.csv"


def _today():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _dataset_stamp(path, df, y):
    """Describe the training data in the bundle, provenance included.

    If the dataset ships a <name>.provenance.json (written by
    generate_cardiac_dataset.py) it is copied in verbatim, so a bundle
    trained on synthetic data carries that disclosure with it and nobody
    downstream has to go looking for it. A dataset with no provenance file
    is stamped "unknown" rather than silently left blank - see
    DATASET_PROVENANCE.md for why that distinction matters here.
    """
    stamp = {
        "file": os.path.basename(path),
        "rows": int(len(df)),
        "sha256": _dataset_sha256(path),
        "n_classes": int(len(np.unique(y))),
    }
    prov_path = path + ".provenance.json"
    if os.path.exists(prov_path):
        with open(prov_path, "r", encoding="utf-8") as f:
            stamp["provenance"] = json.load(f)
    else:
        stamp["provenance"] = {
            "synthetic": "unknown",
            "disclosure": ("No provenance file accompanies this dataset; how it "
                           "was produced is not recorded. Do not cite it as data "
                           "of known origin."),
        }
    return stamp


def _dataset_sha256(path):
    """Hash the training CSV so a bundle can prove which data produced it."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
# Anchored to the project folder: training must overwrite the bundle the
# predictors and the GUI actually load, not create a second copy in
# whatever directory the run was started from - two bundles that disagree
# is precisely the drift this project's shared pipeline exists to prevent.
MODEL_DIR = project_path("triage_model_embedding_english")

# TODO(Sir - open question 1 in ARCHITECTURE.md section 6): standardize the
# embedding model. This is the safe default already used by
# Chosen for being multilingual, small, CPU-friendly and 384 dims.
# The whiteboard's "364" was a placeholder; this model outputs 384.
# Imported from triage_pipeline so training always embeds with the same
# SBERT checkpoint the deployed bundle records in its manifest.
DEFAULT_MODEL = EMBEDDING_MODEL_NAME

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def evaluate(y_true, pred):
    """Accuracy, over-triage % and under-triage % on the test set.

    Classes: 0 = most urgent ... 3 = least urgent.
    Predicting LOWER than truth  = more urgent = over-triage.
    Predicting HIGHER than truth = less urgent = under-triage (dangerous).
    """
    acc = accuracy_score(y_true, pred)
    # Class count comes from the data, not a constant: a cardiac-only dataset
    # can legitimately carry 3 levels, and forcing labels=[0,1,2,3] then
    # reported a phantom empty Level 4 row in every confusion matrix.
    n_classes = int(max(np.max(y_true), np.max(pred))) + 1
    cm = confusion_matrix(y_true, pred, labels=list(range(n_classes)))
    total = cm.sum()
    over = sum(cm[t, p] for t in range(n_classes) for p in range(t))
    under = sum(cm[t, p] for t in range(n_classes) for p in range(t + 1, n_classes))
    return acc, 100 * over / total, 100 * under / total, cm


def safety_grade(under_rate):
    if under_rate < 5:  return "A+"
    if under_rate < 10: return "A"
    if under_rate < 15: return "B"
    if under_rate < 20: return "C"
    return "F"


def efficiency_grade(over_rate):
    if over_rate < 15: return "A"
    if over_rate < 25: return "B"
    if over_rate < 35: return "C"
    return "D"


def train_and_score(X_train, X_test, y_train, y_test, label):
    """Train the shared classifier on one text representation and score it."""
    model = LogisticRegression(max_iter=1200, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc, over, under, cm = evaluate(y_test, pred)
    print(f"  {label:<40} accuracy={acc * 100:6.2f}%   "
          f"under-triage={under:5.2f}% ({safety_grade(under)})   "
          f"over-triage={over:5.2f}%")
    return {
        "method": label,
        "accuracy": round(acc * 100, 2),
        "under_triage_pct": round(under, 2),
        "over_triage_pct": round(over, 2),
        "safety_grade": safety_grade(under),
        "efficiency_grade": efficiency_grade(over),
        "n_features": int(X_train.shape[1]),
        "_model": model,
        "_cm": cm,
        "_pred": pred,
    }


# ----------------------------------------------------------------------
# Text representations
# ----------------------------------------------------------------------

def load_embedding_model(model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n[x] 'sentence-transformers' is not installed.")
        print("    Install it once (needs internet), then re-run:")
        print("        pip install -r requirements.txt\n")
        sys.exit(1)

    print(f"\nLoading embedding model: {model_name}")
    print("(first run downloads it; later runs are fully offline)")
    try:
        return SentenceTransformer(model_name, device="cpu")
    except Exception as e:
        print("\n[x] Could not load the embedding model. The first run needs")
        print(f"    internet to download it once. Underlying error:\n    {e}\n")
        sys.exit(1)


def encode(model, texts, tag):
    emb = model.encode(list(texts), batch_size=32, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True)
    print(f"  [ok] {tag:<34} {emb.shape[0]} complaints x {emb.shape[1]} dims")
    return emb


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train the embedding -> fuse -> classify triage pipeline.")
    parser.add_argument("--data", default=DATA_FILE)
    parser.add_argument("--text-column", default="English_Translation",
                        help="column holding the complaint text (the "
                             "translation experiment points this at "
                             "English_Translation)")
    parser.add_argument("--skip-normalization", action=argparse.BooleanOptionalAction, default=True,
                        help="feed text to the model as written, skipping "
                             "the Roman Urdu dictionary. For the English arm.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="sentence-transformers model name")
    parser.add_argument("--out-dir", default=MODEL_DIR)
    parser.add_argument("--results-file", help="Comparison CSV; defaults to the output bundle.")
    parser.add_argument("--deploy", default="C", choices=["auto", "B", "C"],
                        help="Embedding configuration to save: raw B, preprocessed C, or auto.")
    args = parser.parse_args()
    results_file = args.results_file or os.path.join(args.out_dir, "embedding_pipeline_results.csv")

    print("=" * 78)
    print("EMBEDDING TRIAGE PIPELINE  -  train, compare, keep the best")
    print("=" * 78)

    data_path = resolve_project_file(args.data)
    df = pd.read_csv(data_path)
    # --text-column lets the translation experiment train on the English
    # column of the same file, so both arms share identical vitals, labels
    # and split and differ ONLY in the text. Default is unchanged.
    if args.text_column != "Complaint_Text":
        if args.text_column not in df.columns:
            raise SystemExit(f"--text-column {args.text_column!r} not in "
                             f"{data_path}. Available: {list(df.columns)}")
        df["Complaint_Text"] = df[args.text_column]
        print(f"[ok] Using text column {args.text_column!r} as the complaint text")
    df["Complaint_Text"] = df["Complaint_Text"].fillna("").astype(str)
    print(f"[ok] Loaded {len(df)} patients from {data_path}")

    y = df["Triage_Level"].values - 1          # dataset 1..4 -> classes 0..3
    raw_texts = df["Complaint_Text"].tolist()

    # ---- structured features (identical to the published system) ----
    for enc_col, src in [("Gender_enc", "Gender"), ("Mode_enc", "Mode_of_Arrival"),
                         ("AVPU_enc", "AVPU"), ("ECG_enc", "ECG_Status")]:
        df[enc_col] = LabelEncoder().fit_transform(df[src].astype(str))
    encoders = {
        "gender_enc": LabelEncoder().fit(df["Gender"].astype(str)),
        "mode_enc":   LabelEncoder().fit(df["Mode_of_Arrival"].astype(str)),
        "avpu_enc":   LabelEncoder().fit(df["AVPU"].astype(str)),
        "ecg_enc":    LabelEncoder().fit(df["ECG_Status"].astype(str)),
    }
    for col in NUMERICAL_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    scaler = StandardScaler().fit(df[NUMERICAL_FEATURES])
    df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])

    # Categoricals are ONE-HOT, not the raw LabelEncoder integers. Those
    # integers are alphabetical codes with no ordering, and feeding them to a
    # linear model made it read "ST elevation" as ten times "Abnormal". See
    # encode_categoricals() in triage_pipeline.py for the measured cost.
    # The codes are built through the SAME saved encoders inference uses, so
    # the two paths cannot drift apart.
    cat_codes = np.column_stack([
        encoders["gender_enc"].transform(df["Gender"].astype(str)),
        encoders["mode_enc"].transform(df["Mode_of_Arrival"].astype(str)),
        encoders["avpu_enc"].transform(df["AVPU"].astype(str)),
        encoders["ecg_enc"].transform(df["ECG_Status"].astype(str)),
    ])
    cat_onehot = encode_categoricals(
        {"manifest": {"categorical_encoding": "onehot"},
         "le_gender": encoders["gender_enc"], "le_mode": encoders["mode_enc"],
         "le_avpu": encoders["avpu_enc"], "le_ecg": encoders["ecg_enc"]},
        cat_codes)
    structured = np.hstack([df[NUMERICAL_FEATURES].values, cat_onehot])
    print(f"[ok] Structured block: {len(NUMERICAL_FEATURES)} numeric + "
          f"{cat_onehot.shape[1]} one-hot categorical = {structured.shape[1]} dims")

    # ---- one split, reused by every representation ----
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y)
    y_train, y_test = y[idx_train], y[idx_test]
    print(f"[ok] Split: {len(idx_train)} train / {len(idx_test)} test "
          f"(stratified, random_state={RANDOM_STATE})")

    # ==================================================================
    # CONTRIBUTION 1 - learn stop words on the TRAINING SPLIT ONLY
    # ==================================================================
    print("\n" + "=" * 78)
    print("STEP 1  -  AUTOMATIC STOP-WORD LEARNING  (Contribution 1)")
    print("=" * 78)

    # --skip-normalization exists for the English arm of the translation
    # experiment. Running English through normalize_roman_urdu does not
    # leave it alone - the dictionary is effectively bidirectional, so
    # "chest pain" comes out as "sēna dárd". That is a perfectly reasonable
    # thing to do, but it is not an English pipeline, and a comparison that
    # quietly translated English back into Roman Urdu canonicals would be
    # measuring the dictionary twice instead of comparing two languages.
    if args.skip_normalization:
        print("[note] --skip-normalization: text goes to the model as written")
        normalized = list(raw_texts)
    else:
        normalized = [normalize_roman_urdu(t) for t in raw_texts]
    train_stops, train_report = learn_stopwords(
        [normalized[i] for i in idx_train],
        [y[i] + 1 for i in idx_train],
    )
    print(summarize(train_report))
    print("\n  (learned on the training split only - using all patients "
          "here\n   would leak test labels into preprocessing and inflate the score)")

    # ==================================================================
    # STEP 2 - preprocessing:  clean -> fuzzy -> remove stop words
    # ==================================================================
    print("\n" + "=" * 78)
    print("STEP 2  -  PREPROCESSING  (clean -> fuzzy -> stop-word removal)")
    print("=" * 78)

    if args.skip_normalization:
        from stopwords import remove_stopwords
        preprocessed = [remove_stopwords(t, train_stops) for t in normalized]
    else:
        preprocessed = preprocess_corpus_for_embedding(raw_texts, train_stops)
    n_changed = sum(1 for a, b in zip(normalized, preprocessed) if a != b)
    print(f"  complaints affected by stop-word removal : {n_changed} / {len(df)}")
    print(f"  example raw          : {raw_texts[0]}")
    print(f"  example normalized   : {normalized[0]}")
    print(f"  example preprocessed : {preprocessed[0]}")

    # ==================================================================
    # STEP 3 - build every text representation and compare
    # ==================================================================
    print("\n" + "=" * 78)
    print("STEP 3  -  TEXT REPRESENTATIONS")
    print("=" * 78)

    split = lambda arr: (np.asarray(arr)[idx_train], np.asarray(arr)[idx_test])
    struct_train, struct_test = structured[idx_train], structured[idx_test]
    emb_model = load_embedding_model(args.model)
    emb_raw = encode(emb_model, raw_texts, "Embeddings (raw text)")
    emb_prep = encode(emb_model, preprocessed, "Embeddings (preprocessed)")
    er_train, er_test = split(emb_raw)
    ep_train, ep_test = split(emb_prep)
    candidates = [
        ("B) Embeddings, raw text",
         np.hstack([struct_train, er_train]), np.hstack([struct_test, er_test]),
         {"text_representation": "embeddings_raw"}),
        ("C) Embeddings + preprocessing",
         np.hstack([struct_train, ep_train]), np.hstack([struct_test, ep_test]),
         {"text_representation": "embeddings_preprocessed"}),
    ]

    print("\n" + "=" * 78)
    print("STEP 4  -  RESULTS  (same split, same vitals, same classifier)")
    print("=" * 78)

    results = []
    for label, Xtr, Xte, meta in candidates:
        r = train_and_score(Xtr, Xte, y_train, y_test, label)
        r.update(meta)
        results.append(r)

    # ---- did Contribution 1 help the embeddings? ----
    by_rep = {r["text_representation"]: r for r in results}
    if "embeddings_raw" in by_rep and "embeddings_preprocessed" in by_rep:
        raw_r, prep_r = by_rep["embeddings_raw"], by_rep["embeddings_preprocessed"]
        d_acc = prep_r["accuracy"] - raw_r["accuracy"]
        d_under = prep_r["under_triage_pct"] - raw_r["under_triage_pct"]
        print("\n" + "-" * 78)
        print("ABLATION - effect of automatic stop-word removal on the embeddings")
        print("-" * 78)
        print(f"  accuracy     {raw_r['accuracy']:.2f}%  ->  {prep_r['accuracy']:.2f}%"
              f"   ({d_acc:+.2f} points)")
        print(f"  under-triage {raw_r['under_triage_pct']:.2f}%  ->  "
              f"{prep_r['under_triage_pct']:.2f}%   ({d_under:+.2f} points)")
        if d_acc > 0 or d_under < 0:
            print("  -> preprocessing helped the embedding path.")
        elif d_acc == 0 and d_under == 0:
            print("  -> no measurable change on this dataset.")
        else:
            print("  -> preprocessing did NOT help here; report this honestly.")

    # ==================================================================
    # SELECTION RULE: within 0.5 points of minimum under-triage, then accuracy.
    SAFETY_TOLERANCE = 0.5   # percentage points of under-triage

    most_accurate = max(results, key=lambda r: (r["accuracy"], -r["under_triage_pct"]))
    min_under = min(r["under_triage_pct"] for r in results)
    safety_equivalent = [r for r in results
                         if r["under_triage_pct"] <= min_under + SAFETY_TOLERANCE]
    auto_best = max(safety_equivalent,
                    key=lambda r: (r["accuracy"], -r["under_triage_pct"]))

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  Most accurate overall : {most_accurate['method']}  "
          f"({most_accurate['accuracy']:.2f}%, "
          f"under-triage {most_accurate['under_triage_pct']:.2f}%)")
    print(f"  Safety-first pick     : {auto_best['method']}  "
          f"({auto_best['accuracy']:.2f}%, "
          f"under-triage {auto_best['under_triage_pct']:.2f}%)")

    if most_accurate is not auto_best:
        print(f"  The safety rule rejects the most accurate method because its "
              f"under-triage ({most_accurate['under_triage_pct']:.2f}%)")
        print(f"  is more than {SAFETY_TOLERANCE} points worse than the safest "
              f"option ({min_under:.2f}%).")

    # ==================================================================
    # Deployment defaults to the preprocessed embedding configuration.
    by_letter = {r["method"][0]: r for r in results}
    if args.deploy == "auto":
        best = auto_best
        selection_mode = (
            f"auto: lowest under-triage first (within {SAFETY_TOLERANCE} points), "
            "accuracy as tie-breaker")
    else:
        if args.deploy not in by_letter:
            print(f"\n[x] --deploy {args.deploy} is not available in this run "
                  f"(have: {', '.join(sorted(by_letter))}).")
            sys.exit(1)
        best = by_letter[args.deploy]
        selection_mode = (f"explicit --deploy {args.deploy}: the offline embedding "
                          "model is the method this project deploys on purpose")
        if best is not auto_best:
            print(f"\n  OVERRIDE: --deploy {args.deploy} ships "
                  f"{best['method']} instead of the safety-first pick.")
            d = best["under_triage_pct"] - auto_best["under_triage_pct"]
            if d > 0:
                print(f"  Cost of that choice: {d:+.2f} points of under-triage "
                      f"({auto_best['under_triage_pct']:.2f}% -> "
                      f"{best['under_triage_pct']:.2f}%). Stated, not hidden.")

    print(f"\n  SELECTED : {best['method']}")
    print(f"  accuracy {best['accuracy']:.2f}%   "
          f"under-triage {best['under_triage_pct']:.2f}% ({best['safety_grade']})   "
          f"over-triage {best['over_triage_pct']:.2f}% ({best['efficiency_grade']})")

    print("\n  Confusion matrix of the winner (rows = true, cols = predicted):")
    for i, row in enumerate(best["_cm"]):
        print(f"    True L{i}  " + "".join(f"{v:>7}" for v in row))
    print()
    # Only the classes this dataset actually contains. Forcing all four
    # printed a phantom "L3 Non-Urgent" row with zero support on a
    # cardiac-only dataset, and dragged the macro average down with it.
    _all_names = ["L0 Emergency", "L1 Urgent", "L2 Standard", "L3 Non-Urgent"]
    _labels = sorted(set(np.unique(y_test)) | set(np.unique(best["_pred"])))
    print(classification_report(
        y_test, best["_pred"], labels=_labels,
        target_names=[_all_names[i] for i in _labels]))

    # ==================================================================
    # STEP 5 - ship the winner
    # ==================================================================
    print("=" * 78)
    print("STEP 5  -  SAVING THE WINNING PIPELINE")
    print("=" * 78)

    os.makedirs(args.out_dir, exist_ok=True)

    # Re-learn stop words on ALL labelled data for the shipped artifact:
    # the leak-free split above was for honest measurement, but the model
    # that goes to production should use every patient available.
    final_stops, final_report = learn_stopwords(normalized, df["Triage_Level"].tolist())
    final_report["note"] = (
        "Learned on the full dataset for deployment. The evaluation in "
        "train_embedding_pipeline.py uses a training-split-only list to "
        "avoid label leakage; see 'training_split_stopwords' for that list."
    )
    final_report["training_split_stopwords"] = sorted(train_stops)
    print(f"  [ok] learned_stopwords.json  "
          f"({final_report['n_stopwords']} stop words, full dataset)")
    only_full = sorted(final_stops - train_stops)
    only_train = sorted(train_stops - final_stops)
    if only_full or only_train:
        print(f"       full-data only : {only_full or '-'}")
        print(f"       train-only     : {only_train or '-'}")

    rep = best["text_representation"]
    uses_emb = True
    joblib.dump(best["_model"], os.path.join(args.out_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.pkl"))
    for name, enc in encoders.items():
        joblib.dump(enc, os.path.join(args.out_dir, f"{name}.pkl"))
    save_stopwords(final_report, os.path.join(args.out_dir, "learned_stopwords.json"))
    embedding_model_name = args.model
    embedding_dim = int(emb_model.get_sentence_embedding_dimension())

    # The manifest is how every predictor knows what this directory holds.
    # See triage_pipeline.read_manifest / build_text_features.
    manifest = {
        "method": best["method"],
        "text_representation": rep,
        "embedding_model": embedding_model_name,
        "embedding_dim": embedding_dim,
        "text_pipeline": (
            "raw text -> sentence-transformer" if rep == "embeddings_raw" else
            ("English text" if args.skip_normalization else "clean -> rule replace -> fuzzy -> diacritize")
            + " -> learned stop-word removal -> sentence-transformer"),
        "feature_blocks": [
            {"name": "structured", "dim": int(structured.shape[1])},
            {"name": "embedding", "dim": embedding_dim, "rescaled": False}],
        "trained_by": "train_embedding_pipeline.py",
        "deploy_mode": args.deploy,
        # Read by encode_categoricals() at inference. Bundles saved before
        # this key existed are treated as "ordinal" and keep working.
        "categorical_encoding": "onehot",
        # Inference must reproduce this exactly - see build_text_features.
        "skip_normalization": bool(args.skip_normalization),
        "text_column": args.text_column,
        "dataset": _dataset_stamp(resolve_project_file(args.data), df, y),
        "scope": {
            "clinical_scope": "CARDIAC COMPLAINTS ONLY",
            "note": ("Trained exclusively on cardiac presentations. Predictions "
                     "for non-cardiac complaints (trauma, burns, neurological, "
                     "obstetric, etc.) are OUT OF SCOPE and must not be relied "
                     "on."),
            "categories_in_training_data": sorted(
                df["Category"].astype(str).unique().tolist())
            if "Category" in df else [],
        },
        "trained_on_date": _today(),
    }
    with open(os.path.join(args.out_dir, "model_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    metrics = {
        "winning_method": best["method"],
        "deployed_method": best["method"],
        "selection_rule": selection_mode,
        "safety_first_pick": auto_best["method"],
        "deploy_override_used": args.deploy != "auto" and best is not auto_best,
        "most_accurate_method": most_accurate["method"],
        "most_accurate_not_selected": most_accurate["method"] != best["method"],
        "text_representation": rep,
        "uses_embeddings": uses_emb,
        "embedding_model": embedding_model_name,
        "embedding_dim": embedding_dim,
        "accuracy": best["accuracy"],
        "under_triage_pct": best["under_triage_pct"],
        "over_triage_pct": best["over_triage_pct"],
        "safety_grade": best["safety_grade"],
        "efficiency_grade": best["efficiency_grade"],
        "n_features": best["n_features"],
        "confusion_matrix": best["_cm"].tolist(),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "random_state": RANDOM_STATE,
        "preprocessing": "clean -> fuzzy normalize -> learned stop-word removal",
        "n_stopwords_deployed": final_report["n_stopwords"],
        "all_methods": [{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in results],
    }
    with open(os.path.join(args.out_dir, "triage_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                  for r in results]).to_csv(results_file, index=False)

    print(f"  [ok] Model + encoders saved to '{args.out_dir}/'")
    print(f"  [ok] Manifest saved to '{args.out_dir}/model_manifest.json'")
    print(f"  [ok] Metrics saved to '{args.out_dir}/triage_metrics.json'")
    print(f"  [ok] Comparison table saved to '{results_file}'")
    print(f"\n  DEPLOYED: {best['method']}")
    print(f"            text representation : {rep}")
    print(f"            uses embeddings     : {'YES - ' + str(embedding_model_name) if uses_emb else 'NO'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
