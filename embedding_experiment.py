"""
embedding_experiment.py
=======================================================================
Does an offline AI embedding model actually help our Roman Urdu triage?

This script answers Sir's question with NUMBERS instead of a guess. It
trains the SAME classifier (Logistic Regression) on the SAME train/test
split with the SAME vital-sign features, and changes ONLY how the
complaint text is turned into numbers. It compares three text methods:

  A) DICTIONARY + BoW  -> what we use now (fuzzy match + diacritization
                          + word/char Bag-of-Words + attention weights)
  B) EMBEDDINGS ONLY   -> an offline sentence-embedding AI model turns
                          each complaint into a meaning vector
  C) BOTH COMBINED     -> dictionary features AND embeddings together

It prints an accuracy + safety comparison, and a small "synonym test"
that shows whether Roman Urdu words like dard / drd / pain actually land
close together in the embedding space.

-----------------------------------------------------------------------
HOW TO RUN
-----------------------------------------------------------------------
1. Install the extra library (needs internet ONCE to download it and the
   model; after that it runs fully offline):

       pip install -r requirements-embedding.txt

2. Run:

       python embedding_experiment.py

   Optionally pick a different embedding model:

       python embedding_experiment.py --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

The first run downloads the model (a few hundred MB) into a local cache.
Every run after that works with no internet.
=======================================================================
"""

import os
import sys
import argparse
import warnings

# See prediction.py: keeps `import triage_pipeline` working regardless of
# how this script was launched.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from triage_pipeline import (
    normalize_roman_urdu,
    build_attention_weights,
    project_path,
    resolve_project_file,
    NUMERICAL_FEATURES,
)

warnings.filterwarnings("ignore")

DATA_FILE = "triage_mixed_language_dataset.csv"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ----------------------------------------------------------------------
# Metrics (same definition the main training script uses)
# ----------------------------------------------------------------------
def evaluate(y_test, pred):
    """Return accuracy, over-triage %, under-triage % from the test set.

    Classes: 0 = most urgent ... 3 = least urgent.
    Predicting a LOWER number than truth  = MORE urgent = over-triage.
    Predicting a HIGHER number than truth = LESS urgent = under-triage (dangerous).
    """
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])
    total = cm.sum()
    over = sum(cm[t, p] for t in range(4) for p in range(t))
    under = sum(cm[t, p] for t in range(4) for p in range(t + 1, 4))
    over_rate = 100 * over / total if total else 0
    under_rate = 100 * under / total if total else 0
    return acc, over_rate, under_rate


def safety_grade(under_rate):
    if under_rate < 5:  return "A+"
    if under_rate < 10: return "A"
    if under_rate < 15: return "B"
    if under_rate < 20: return "C"
    return "F"


def train_and_score(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = LogisticRegression(max_iter=1200, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc, over, under = evaluate(y_test, pred)
    print(f"  [ok] {label:<28} accuracy={acc*100:5.2f}%   "
          f"under-triage={under:4.1f}% ({safety_grade(under)})   "
          f"over-triage={over:4.1f}%")
    return {"method": label, "accuracy": round(acc * 100, 2),
            "under_triage_pct": round(under, 2),
            "over_triage_pct": round(over, 2),
            "safety_grade": safety_grade(under)}


# ----------------------------------------------------------------------
# Text representation A: dictionary + Bag of Words + attention
# (identical to triage_bow_fuzzy_diac.py)
# ----------------------------------------------------------------------
def build_dictionary_bow_features(df):
    complaints = df["Complaint_Text_norm"].astype(str)
    word_bow = CountVectorizer(max_features=350, ngram_range=(1, 2), analyzer="word")
    char_bow = CountVectorizer(max_features=350, ngram_range=(2, 4), analyzer="char_wb")
    word_features = word_bow.fit_transform(complaints).toarray()
    char_features = char_bow.fit_transform(complaints).toarray()
    text_features = np.hstack([word_features, char_features])
    feature_names = (list(word_bow.get_feature_names_out()) +
                     list(char_bow.get_feature_names_out()))
    weights = build_attention_weights(feature_names)
    return text_features * weights


# ----------------------------------------------------------------------
# Text representation B: offline embedding model
# ----------------------------------------------------------------------
def build_embedding_features(texts, model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n[x] The 'sentence-transformers' library is not installed.")
        print("    Install it once (needs internet), then re-run:")
        print("        pip install -r requirements-embedding.txt\n")
        sys.exit(1)

    print(f"\nLoading embedding model: {model_name}")
    print("(first run downloads it; later runs are fully offline)")
    try:
        model = SentenceTransformer(model_name, device="cpu")
    except Exception as e:
        print(f"\n[x] Could not load the model. If this is the first run you need")
        print(f"    internet to download it once. Underlying error:\n    {e}\n")
        sys.exit(1)
    emb = model.encode(list(texts), batch_size=32, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    print(f"[ok] Embeddings ready: {emb.shape[0]} patients x {emb.shape[1]} dims")
    return emb, model


# ----------------------------------------------------------------------
# Synonym clustering sanity check
# ----------------------------------------------------------------------
def synonym_check(model):
    """Show whether meaning-equivalent words land close in embedding space.
    Cosine similarity ranges -1..1; higher = more similar."""
    groups = {
        "chest pain": ["seena mein dard", "seene mein drd", "chest pain"],
        "fever":      ["bukhar", "bukhaar", "fever"],
        "breathless": ["saans phoolna", "saans nahi aa rahi", "shortness of breath"],
    }
    print("\n" + "=" * 70)
    print("SYNONYM TEST  (do same-meaning words land close together?)")
    print("=" * 70)
    print("Cosine similarity: 1.0 = identical meaning, 0 = unrelated.\n")
    all_texts, labels = [], []
    for g, items in groups.items():
        for it in items:
            all_texts.append(it); labels.append(g)
    vecs = model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)

    # within-group similarity (should be HIGH) vs across-group (should be LOW)
    import itertools
    within, across = [], []
    for i, j in itertools.combinations(range(len(all_texts)), 2):
        sim = float(np.dot(vecs[i], vecs[j]))
        (within if labels[i] == labels[j] else across).append(sim)
    for g, items in groups.items():
        idx = [k for k, l in enumerate(labels) if l == g]
        print(f"  {g}:")
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                s = float(np.dot(vecs[idx[a]], vecs[idx[b]]))
                print(f"      '{all_texts[idx[a]]}'  <->  '{all_texts[idx[b]]}'"
                      f"   sim = {s:.2f}")
    print(f"\n  AVERAGE same-meaning similarity : {np.mean(within):.2f}  (want HIGH)")
    print(f"  AVERAGE different-meaning sim   : {np.mean(across):.2f}  (want LOW)")
    gap = np.mean(within) - np.mean(across)
    print(f"  Separation gap                  : {gap:.2f}", end="  ")
    print("-> good, meanings separate" if gap > 0.15
          else "-> weak; Roman Urdu may not cluster well with this model")


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare dictionary vs embeddings for triage text.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="sentence-transformers model name")
    parser.add_argument("--data", default=DATA_FILE, help="dataset CSV")
    args = parser.parse_args()

    print("=" * 70)
    print("EMBEDDING vs DICTIONARY  -  TRIAGE TEXT REPRESENTATION EXPERIMENT")
    print("=" * 70)

    data_path = resolve_project_file(args.data)
    df = pd.read_csv(data_path)
    print(f"[ok] Loaded {len(df)} patients from {data_path}")

    # Normalized text (for dictionary path) and raw text (for embeddings)
    df["Complaint_Text"] = df["Complaint_Text"].fillna("")
    df["Complaint_Text_norm"] = df["Complaint_Text"].apply(normalize_roman_urdu)
    raw_texts = df["Complaint_Text"].astype(str).tolist()

    # ---- shared structured features (identical to training) ----
    for enc_col, src in [("Gender_enc", "Gender"), ("Mode_enc", "Mode_of_Arrival"),
                         ("AVPU_enc", "AVPU"), ("ECG_enc", "ECG_Status")]:
        df[enc_col] = LabelEncoder().fit_transform(df[src].astype(str))
    for col in NUMERICAL_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    df[NUMERICAL_FEATURES] = StandardScaler().fit_transform(df[NUMERICAL_FEATURES])
    structured = df[NUMERICAL_FEATURES +
                    ["Gender_enc", "Mode_enc", "AVPU_enc", "ECG_enc"]].values

    y = df["Triage_Level"].values - 1

    # ---- build the three text representations ----
    print("\nBuilding text representations...")
    dict_feats = build_dictionary_bow_features(df)
    print(f"  [ok] Dictionary+BoW features: {dict_feats.shape[1]} dims")
    emb_feats, model = build_embedding_features(raw_texts, args.model)

    # ---- assemble full feature matrices (text + same vitals) ----
    X_dict = np.hstack([structured, dict_feats])
    X_emb  = np.hstack([structured, emb_feats])
    X_both = np.hstack([structured, dict_feats, emb_feats])

    # ---- train + score all three ----
    print("\n" + "=" * 70)
    print("RESULTS  (same split, same vitals, same classifier)")
    print("=" * 70)
    results = []
    results.append(train_and_score(X_dict, y, "A) Dictionary + BoW (current)"))
    results.append(train_and_score(X_emb,  y, "B) Embeddings only"))
    results.append(train_and_score(X_both, y, "C) Both combined"))

    # ---- verdict ----
    best = max(results, key=lambda r: (r["accuracy"], -r["under_triage_pct"]))
    baseline = results[0]["accuracy"]
    print("\n" + "-" * 70)
    print(f"Best method: {best['method']}  ({best['accuracy']}% accuracy, "
          f"{best['under_triage_pct']}% under-triage)")
    delta = best["accuracy"] - baseline
    if best["method"].startswith("A"):
        print("Embeddings did NOT beat the dictionary on this data -> keep the dictionary.")
    elif delta < 0.5:
        print(f"Embeddings barely changed accuracy ({delta:+.2f}%) -> dictionary is fine; "
              "the combined option is only worth it if under-triage drops.")
    else:
        print(f"Embeddings improved accuracy by {delta:+.2f}% -> worth adopting "
              f"({best['method']}).")

    # ---- save results + synonym check ----
    results_path = project_path("embedding_experiment_results.csv")
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"\n[ok] Results saved to '{results_path}'")

    synonym_check(model)
    print("\nDone.")


if __name__ == "__main__":
    main()
