"""
train_embedding_pipeline.py
=======================================================================
TASK 3 - Embedding -> Fuse -> Classifier, as the main training flow.
=======================================================================

Implements the target architecture from ARCHITECTURE.md section 1:

    structured (age, vitals, ECG, AVPU) ----------------+
                                                        |
    complaint text                                      |
      -> clean -> fuzzy normalize                       |
      -> remove LEARNED stop words (Contribution 1)     |
      -> Sentence Transformer -> embedding vector       |
                                                        v
                                                    [ FUSE ]
                                                        |
                                                        v
                                          Logistic Regression -> triage 1-4

It also re-runs the three-way comparison (dictionary / embeddings /
both) that ARCHITECTURE.md Task 3 asks for, now that automatic
stop-word removal sits in front of the embedding step, and keeps
whichever representation scores best on accuracy and under-triage.

-----------------------------------------------------------------------
ABLATION - does Contribution 1 actually help the embeddings?
-----------------------------------------------------------------------
The comparison deliberately includes embeddings computed on RAW text and
embeddings computed on PREPROCESSED text. The difference between those
two rows is the measured effect of the automatic stop-word learner on
the embedding path, which is the number the paper needs.

    A)  Dictionary + BoW + attention        (current published system)
    B)  Embeddings, raw text                (no preprocessing)
    C)  Embeddings, preprocessed text       (fuzzy + learned stop words)
    D)  Dictionary + Embeddings, hybrid     (preprocessed)

-----------------------------------------------------------------------
NO LABEL LEAKAGE
-----------------------------------------------------------------------
The stop-word learner uses the triage LABELS, so learning it on all
1,204 patients and then scoring on a held-out split would leak test
labels into preprocessing and inflate the result. This script therefore:

  * learns stop words on the TRAINING SPLIT ONLY for the evaluation, and
  * re-learns them on the FULL dataset for the shipped artifact, once
    the winning representation has been chosen.

Both lists are reported so the difference is visible.

-----------------------------------------------------------------------
WHY A SEPARATE MODEL DIRECTORY
-----------------------------------------------------------------------
Artifacts go to `triage_model_embedding/`, NOT `triage_model/`. The
existing predictors (predict_batch.py, prediction.py,
prediction_interactive.py) load `triage_model/` and must keep working
unchanged; overwriting it with an embedding-based model whose feature
layout is different would break all three.

-----------------------------------------------------------------------
HOW TO RUN
-----------------------------------------------------------------------
    pip install -r requirements-embedding.txt     # once, needs internet
    python train_embedding_pipeline.py

    python train_embedding_pipeline.py --model <other-sentence-transformer>
    python train_embedding_pipeline.py --skip-embeddings   # dictionary only

The first run downloads the embedding model (a few hundred MB) into the
local Hugging Face cache; every run afterwards is fully offline.
=======================================================================
"""

import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from stopwords import learn_stopwords, save_stopwords, summarize
from triage_pipeline import (
    NUMERICAL_FEATURES,
    build_attention_weights,
    make_console_safe,
    normalize_roman_urdu,
    preprocess_corpus_for_embedding,
)

warnings.filterwarnings("ignore")
make_console_safe()

DATA_FILE = "triage_mixed_language_dataset.csv"
MODEL_DIR = "triage_model_embedding"
RESULTS_FILE = "embedding_pipeline_results.csv"

# TODO(Sir - open question 1 in ARCHITECTURE.md section 6): standardize the
# embedding model. This is the safe default already used by
# embedding_experiment.py - multilingual, small, CPU-friendly, 384 dims.
# The whiteboard's "364" was a placeholder; this model outputs 384.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ----------------------------------------------------------------------
# Metrics (identical definitions to triage_bow_fuzzy_diac.py)
# ----------------------------------------------------------------------

def evaluate(y_true, pred):
    """Accuracy, over-triage % and under-triage % on the test set.

    Classes: 0 = most urgent ... 3 = least urgent.
    Predicting LOWER than truth  = more urgent = over-triage.
    Predicting HIGHER than truth = less urgent = under-triage (dangerous).
    """
    acc = accuracy_score(y_true, pred)
    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2, 3])
    total = cm.sum()
    over = sum(cm[t, p] for t in range(4) for p in range(t))
    under = sum(cm[t, p] for t in range(4) for p in range(t + 1, 4))
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

def build_bow(train_texts, test_texts):
    """Dictionary + word/char Bag-of-Words + domain-guided attention.

    Fitted on the TRAINING texts only, then applied to the test texts -
    the vectorizers are part of the model, not of the data.
    """
    word_bow = CountVectorizer(max_features=350, ngram_range=(1, 2), analyzer="word")
    char_bow = CountVectorizer(max_features=350, ngram_range=(2, 4), analyzer="char_wb")

    train_mat = np.hstack([
        word_bow.fit_transform(train_texts).toarray(),
        char_bow.fit_transform(train_texts).toarray(),
    ])
    test_mat = np.hstack([
        word_bow.transform(test_texts).toarray(),
        char_bow.transform(test_texts).toarray(),
    ])

    names = (list(word_bow.get_feature_names_out()) +
             list(char_bow.get_feature_names_out()))
    weights = build_attention_weights(names)
    return train_mat * weights, test_mat * weights, word_bow, char_bow


def load_embedding_model(model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n[x] 'sentence-transformers' is not installed.")
        print("    Install it once (needs internet), then re-run:")
        print("        pip install -r requirements-embedding.txt\n")
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
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="sentence-transformers model name")
    parser.add_argument("--out-dir", default=MODEL_DIR)
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="dictionary path only (no model download needed)")
    args = parser.parse_args()

    print("=" * 78)
    print("EMBEDDING TRIAGE PIPELINE  -  train, compare, keep the best")
    print("=" * 78)

    df = pd.read_csv(args.data)
    df["Complaint_Text"] = df["Complaint_Text"].fillna("").astype(str)
    print(f"[ok] Loaded {len(df)} patients from {args.data}")

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
    structured = df[NUMERICAL_FEATURES +
                    ["Gender_enc", "Mode_enc", "AVPU_enc", "ECG_enc"]].values

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

    normalized = [normalize_roman_urdu(t) for t in raw_texts]
    train_stops, train_report = learn_stopwords(
        [normalized[i] for i in idx_train],
        [y[i] + 1 for i in idx_train],
    )
    print(summarize(train_report))
    print("\n  (learned on the training split only - using all 1,204 patients "
          "here\n   would leak test labels into preprocessing and inflate the score)")

    # ==================================================================
    # STEP 2 - preprocessing:  clean -> fuzzy -> remove stop words
    # ==================================================================
    print("\n" + "=" * 78)
    print("STEP 2  -  PREPROCESSING  (clean -> fuzzy -> stop-word removal)")
    print("=" * 78)

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
    norm_train, norm_test = split(normalized)
    prep_train, prep_test = split(preprocessed)
    struct_train, struct_test = structured[idx_train], structured[idx_test]

    bow_train, bow_test, word_bow, char_bow = build_bow(norm_train, norm_test)
    print(f"  [ok] {'Dictionary + BoW + attention':<34} {bow_train.shape[1]} dims")

    candidates = [(
        "A) Dictionary + BoW (current)",
        np.hstack([struct_train, bow_train]),
        np.hstack([struct_test, bow_test]),
        {"text_representation": "dictionary_bow"},
    )]

    emb_model = None
    if not args.skip_embeddings:
        emb_model = load_embedding_model(args.model)
        emb_raw = encode(emb_model, raw_texts, "Embeddings (raw text)")
        emb_prep = encode(emb_model, preprocessed, "Embeddings (preprocessed)")

        er_train, er_test = emb_raw[idx_train], emb_raw[idx_test]
        ep_train, ep_test = emb_prep[idx_train], emb_prep[idx_test]

        candidates += [
            ("B) Embeddings, raw text",
             np.hstack([struct_train, er_train]),
             np.hstack([struct_test, er_test]),
             {"text_representation": "embeddings_raw"}),
            ("C) Embeddings + preprocessing",
             np.hstack([struct_train, ep_train]),
             np.hstack([struct_test, ep_test]),
             {"text_representation": "embeddings_preprocessed"}),
            ("D) Hybrid: dictionary + embeddings",
             np.hstack([struct_train, bow_train, ep_train]),
             np.hstack([struct_test, bow_test, ep_test]),
             {"text_representation": "hybrid"}),
        ]
    else:
        print("\n[skip] --skip-embeddings set: dictionary path only.")

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
    # SELECTION RULE - safety first, then accuracy
    #
    # ARCHITECTURE.md Task 3 says to keep whichever scores best on
    # "accuracy and under-triage". Those two can disagree, and on this
    # dataset they do. In a triage system they are NOT interchangeable:
    # under-triage means a patient is sent to the back of the queue when
    # they needed immediate care, so it is the error that harms people.
    # Raw accuracy is therefore the tie-breaker, not the objective.
    #
    # Rule: treat every representation whose under-triage is within
    # SAFETY_TOLERANCE points of the lowest under-triage as
    # safety-equivalent, then take the most accurate of those. This also
    # lands on the hybrid option that ARCHITECTURE.md section 4.3 names
    # as the safe default (the dictionary keeps anchoring the Roman Urdu
    # the embedding model misses).
    # ==================================================================
    SAFETY_TOLERANCE = 0.5   # percentage points of under-triage

    baseline = results[0]
    most_accurate = max(results, key=lambda r: (r["accuracy"], -r["under_triage_pct"]))
    min_under = min(r["under_triage_pct"] for r in results)
    safety_equivalent = [r for r in results
                         if r["under_triage_pct"] <= min_under + SAFETY_TOLERANCE]
    best = max(safety_equivalent, key=lambda r: (r["accuracy"], -r["under_triage_pct"]))

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  Most accurate overall : {most_accurate['method']}  "
          f"({most_accurate['accuracy']:.2f}%, "
          f"under-triage {most_accurate['under_triage_pct']:.2f}%)")

    if most_accurate is not best:
        print(f"  NOT selected: its under-triage ({most_accurate['under_triage_pct']:.2f}%) "
              f"is worse than the safest option ({min_under:.2f}%).")
        print("  Under-triage sends a critically ill patient to the back of the")
        print("  queue, so it outranks raw accuracy in the selection rule.")

    print(f"\n  SELECTED : {best['method']}")
    print(f"  accuracy {best['accuracy']:.2f}%   "
          f"under-triage {best['under_triage_pct']:.2f}% ({best['safety_grade']})   "
          f"over-triage {best['over_triage_pct']:.2f}% ({best['efficiency_grade']})")

    d_acc = best["accuracy"] - baseline["accuracy"]
    d_under = best["under_triage_pct"] - baseline["under_triage_pct"]
    if best is baseline:
        print("  The dictionary is still the best text representation on this data.")
    else:
        acc_word = "better" if d_acc >= 0 else "worse"
        if d_under < 0:
            safety_word = f"and {abs(d_under):.2f} points SAFER (less under-triage)"
        elif d_under == 0:
            safety_word = "at identical under-triage"
        else:
            safety_word = f"but {d_under:.2f} points MORE under-triage (worse)"
        print(f"  vs the current dictionary system: {abs(d_acc):.2f} accuracy points "
              f"{acc_word}, {safety_word}.")

    print("\n  Confusion matrix of the winner (rows = true, cols = predicted):")
    for i, row in enumerate(best["_cm"]):
        print(f"    True L{i}  " + "".join(f"{v:>7}" for v in row))
    print()
    print(classification_report(
        y_test, best["_pred"], labels=[0, 1, 2, 3],
        target_names=["L0 Emergency", "L1 Urgent", "L2 Standard", "L3 Non-Urgent"]))

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
    save_stopwords(final_report)
    print(f"  [ok] learned_stopwords.json  "
          f"({final_report['n_stopwords']} stop words, full dataset)")
    only_full = sorted(final_stops - train_stops)
    only_train = sorted(train_stops - final_stops)
    if only_full or only_train:
        print(f"       full-data only : {only_full or '-'}")
        print(f"       train-only     : {only_train or '-'}")

    joblib.dump(best["_model"], os.path.join(args.out_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.pkl"))
    for name, enc in encoders.items():
        joblib.dump(enc, os.path.join(args.out_dir, f"{name}.pkl"))
    if best["text_representation"] in ("dictionary_bow", "hybrid"):
        joblib.dump(word_bow, os.path.join(args.out_dir, "word_bow.pkl"))
        joblib.dump(char_bow, os.path.join(args.out_dir, "char_bow.pkl"))

    metrics = {
        "winning_method": best["method"],
        "selection_rule": (
            f"Lowest under-triage first (within {SAFETY_TOLERANCE} points), "
            "accuracy as tie-breaker. Under-triage outranks accuracy because "
            "it is the error that harms patients."
        ),
        "most_accurate_method": most_accurate["method"],
        "most_accurate_not_selected": most_accurate["method"] != best["method"],
        "text_representation": best["text_representation"],
        "embedding_model": (None if best["text_representation"] == "dictionary_bow"
                            else args.model),
        "embedding_dim": (None if emb_model is None
                          else int(emb_model.get_sentence_embedding_dimension())),
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
                  for r in results]).to_csv(RESULTS_FILE, index=False)

    print(f"  [ok] Model + encoders saved to '{args.out_dir}/'")
    print(f"  [ok] Metrics saved to '{args.out_dir}/triage_metrics.json'")
    print(f"  [ok] Comparison table saved to '{RESULTS_FILE}'")
    print("\n  NOTE: the published dictionary model in 'triage_model/' is left")
    print("        untouched, so predict_batch.py / prediction.py /")
    print("        prediction_interactive.py keep working exactly as before.")
    print("\nDone.")


if __name__ == "__main__":
    main()
