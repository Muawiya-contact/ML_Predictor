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
stop-word removal sits in front of the embedding step, and reports
which representation scores best on accuracy and under-triage.

WHAT SHIPS is an explicit choice: --deploy, default C (offline
embeddings + preprocessing). It used to be decided by the safety rule
alone, which tied between A and D and kept the first - so this script
shipped a dictionary-only model into triage_model_embedding/ while
reporting itself as the embedding pipeline. Two things caused that and
both are fixed here: the tie (see --deploy) and the reason for the tie
(the hybrid's unscaled block concatenation, see STEP 3).

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
predict_batch.py) load `triage_model/` and must keep working
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

# See prediction.py: keeps `import triage_pipeline` working regardless of
# how this script was launched.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

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
    encode_categoricals,
    make_console_safe,
    normalize_roman_urdu,
    preprocess_corpus_for_embedding,
    project_path,
    resolve_project_file,
)

warnings.filterwarnings("ignore")
make_console_safe()

DATA_FILE = "cardiac_multilingual_10000.csv"


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
MODEL_DIR = project_path("triage_model_embedding")
RESULTS_FILE = project_path("embedding_pipeline_results.csv")

# TODO(Sir - open question 1 in ARCHITECTURE.md section 6): standardize the
# embedding model. This is the safe default already used by
# Chosen for being multilingual, small, CPU-friendly and 384 dims.
# The whiteboard's "364" was a placeholder; this model outputs 384.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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
    parser.add_argument("--text-column", default="Complaint_Text",
                        help="column holding the complaint text (the "
                             "translation experiment points this at "
                             "English_Translation)")
    parser.add_argument("--skip-normalization", action="store_true",
                        help="feed text to the model as written, skipping "
                             "the Roman Urdu dictionary. For the English arm.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="sentence-transformers model name")
    parser.add_argument("--out-dir", default=MODEL_DIR)
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="dictionary path only (no model download needed)")
    parser.add_argument("--deploy", default="C", choices=["auto", "A", "B", "C", "D"],
                        help="which method to ship to --out-dir. Default C "
                             "(offline embeddings + preprocessing), the method "
                             "this project deploys. 'auto' uses the safety-first "
                             "rule, which can tie on A and silently ship a "
                             "dictionary-only model.")
    args = parser.parse_args()

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
    print("\n  (learned on the training split only - using all 1,204 patients "
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
    block_scalers = {}
    if not args.skip_embeddings:
        emb_model = load_embedding_model(args.model)
        emb_raw = encode(emb_model, raw_texts, "Embeddings (raw text)")
        emb_prep = encode(emb_model, preprocessed, "Embeddings (preprocessed)")

        er_train, er_test = emb_raw[idx_train], emb_raw[idx_test]
        ep_train, ep_test = emb_prep[idx_train], emb_prep[idx_test]

        # ==============================================================
        # HYBRID FUSION - rescale each block before concatenating.
        #
        # THE BUG THIS FIXES: the BoW block is multiplied by the domain
        # attention weights, so its non-zero entries sit around 1-8, while
        # the embedding block is L2-normalized across 384 dims, so its
        # entries sit around 0.05. Concatenating them raw and fitting one
        # L2-penalised Logistic Regression means a unit of "signal" costs
        # ~100x more coefficient magnitude on the embedding columns than on
        # the BoW columns. The optimiser therefore ignores the embeddings
        # entirely and the hybrid reproduced method A's accuracy,
        # under-triage AND over-triage to the last decimal (90.04 / 3.73 /
        # 6.22), which is what made the "embedding pipeline" ship a
        # dictionary-only model.
        #
        # Standardising each block separately (fit on TRAIN only) puts both
        # on unit variance per column, so the penalty is comparable and the
        # classifier can actually choose between them.
        # ==============================================================
        bow_scaler = StandardScaler().fit(bow_train)
        emb_scaler = StandardScaler().fit(ep_train)
        block_scalers = {"bow": bow_scaler, "embedding": emb_scaler}

        bow_train_s, bow_test_s = (bow_scaler.transform(bow_train),
                                   bow_scaler.transform(bow_test))
        ep_train_s, ep_test_s = (emb_scaler.transform(ep_train),
                                 emb_scaler.transform(ep_test))

        # Report the quantity the L2 penalty actually responds to: the typical
        # spread of a single column. A column with a large spread needs a small
        # coefficient to move the decision, so it is cheap under the penalty; a
        # column with a tiny spread needs a large one and gets shrunk to zero.
        def block_scale(mat):
            nz = np.abs(mat[mat != 0])
            return mat.std(axis=0).mean(), (nz.mean() if nz.size else 0.0)

        (bow_sd, bow_nz), (emb_sd, emb_nz) = block_scale(bow_train), block_scale(ep_train)
        print("\n  block scales before fusion  (why the old hybrid was broken):")
        print(f"    {'':<26}{'mean column sd':>16}{'mean |non-zero|':>18}")
        print(f"    dictionary + BoW block    {bow_sd:>16.4f}{bow_nz:>18.4f}")
        print(f"    embedding block           {emb_sd:>16.4f}{emb_nz:>18.4f}")
        print(f"    ratio                     {bow_sd / max(emb_sd, 1e-12):>15.1f}x"
              f"{bow_nz / max(emb_nz, 1e-12):>17.1f}x")
        print("    A shared L2 penalty across blocks on that ratio means the "
              "embedding\n    columns cost far more coefficient per unit of signal, so the "
              "optimiser\n    dropped them and the hybrid reproduced method A exactly.")
        print(f"    after StandardScaler per block: "
              f"BoW {bow_train_s.std(axis=0).mean():.4f}   "
              f"embedding {ep_train_s.std(axis=0).mean():.4f}")

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
             np.hstack([struct_train, bow_train_s, ep_train_s]),
             np.hstack([struct_test, bow_test_s, ep_test_s]),
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
    # safety-equivalent, then take the most accurate of those.
    #
    # This rule REPORTS a recommendation; it no longer decides what ships.
    # It cannot: when two methods score identically it silently keeps the
    # first, and "first" is the dictionary baseline. See the deploy block
    # below.
    # ==================================================================
    SAFETY_TOLERANCE = 0.5   # percentage points of under-triage

    baseline = results[0]
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
    # WHICH METHOD ACTUALLY SHIPS
    #
    # The safety rule alone is not enough to decide this. On this dataset
    # methods A and D can tie exactly, and `max` keeps the FIRST of a tie -
    # which is A - so an "embedding pipeline" silently deployed a
    # dictionary-only model with embedding_model=null in its metrics.
    # Deployment is therefore an explicit choice, defaulting to the method
    # this project is built to use (C, offline sentence embeddings), with
    # --deploy auto still available to reproduce the old behaviour.
    # ==================================================================
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
            if args.skip_embeddings:
                print("    --skip-embeddings only trains method A.")
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

    if not best["text_representation"].startswith("embeddings") \
            and best["text_representation"] != "hybrid":
        print("\n  [!] WARNING: the method being deployed to "
              f"'{args.out_dir}' does NOT use embeddings.")
        print("      Pass --deploy C to ship the offline embedding model.")

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
    save_stopwords(final_report)
    print(f"  [ok] learned_stopwords.json  "
          f"({final_report['n_stopwords']} stop words, full dataset)")
    only_full = sorted(final_stops - train_stops)
    only_train = sorted(train_stops - final_stops)
    if only_full or only_train:
        print(f"       full-data only : {only_full or '-'}")
        print(f"       train-only     : {only_train or '-'}")

    rep = best["text_representation"]
    uses_bow = rep in ("dictionary_bow", "hybrid")
    uses_emb = rep in ("embeddings_raw", "embeddings_preprocessed", "hybrid")

    joblib.dump(best["_model"], os.path.join(args.out_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.pkl"))
    for name, enc in encoders.items():
        joblib.dump(enc, os.path.join(args.out_dir, f"{name}.pkl"))
    if uses_bow:
        joblib.dump(word_bow, os.path.join(args.out_dir, "word_bow.pkl"))
        joblib.dump(char_bow, os.path.join(args.out_dir, "char_bow.pkl"))

    # Per-block scalers are part of the model: without them at inference time
    # the hybrid would see features on a different scale than it was fitted
    # on. Only the hybrid rescales, so only the hybrid saves them.
    for block, block_scaler in (block_scalers if rep == "hybrid" else {}).items():
        joblib.dump(block_scaler,
                    os.path.join(args.out_dir, f"{block}_block_scaler.pkl"))

    # Stale artifacts from a previous run with a DIFFERENT representation
    # would otherwise be picked up and silently mixed into the feature
    # matrix (e.g. leftover word_bow.pkl from a dictionary run).
    stale = []
    if not uses_bow:
        stale += ["word_bow.pkl", "char_bow.pkl"]
    if rep != "hybrid":
        stale += ["bow_block_scaler.pkl", "embedding_block_scaler.pkl"]
    for name in stale:
        path = os.path.join(args.out_dir, name)
        if os.path.exists(path):
            os.remove(path)
            print(f"  [ok] removed stale artifact from a previous run: {name}")

    embedding_model_name = args.model if uses_emb else None
    embedding_dim = (int(emb_model.get_sentence_embedding_dimension())
                     if (uses_emb and emb_model is not None) else None)

    # The manifest is how every predictor knows what this directory holds.
    # See triage_pipeline.read_manifest / build_text_features.
    manifest = {
        "method": best["method"],
        "text_representation": rep,
        "embedding_model": embedding_model_name,
        "embedding_dim": embedding_dim,
        "text_pipeline": (
            "clean -> rule replace -> fuzzy -> diacritize"
            + (" -> learned stop-word removal" if rep != "embeddings_raw" else "")
            + (" -> BoW + attention" if uses_bow else "")
            + (" -> sentence-transformer" if uses_emb else "")),
        "feature_blocks": (
            [{"name": "structured", "dim": int(structured.shape[1])}]
            + ([{"name": "bow", "dim": int(bow_train.shape[1]),
                 "rescaled": rep == "hybrid"}] if uses_bow else [])
            + ([{"name": "embedding", "dim": embedding_dim,
                 "rescaled": rep == "hybrid"}] if uses_emb else [])),
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
                  for r in results]).to_csv(RESULTS_FILE, index=False)

    print(f"  [ok] Model + encoders saved to '{args.out_dir}/'")
    print(f"  [ok] Manifest saved to '{args.out_dir}/model_manifest.json'")
    print(f"  [ok] Metrics saved to '{args.out_dir}/triage_metrics.json'")
    print(f"  [ok] Comparison table saved to '{RESULTS_FILE}'")
    print(f"\n  DEPLOYED: {best['method']}")
    print(f"            text representation : {rep}")
    print(f"            uses embeddings     : {'YES - ' + str(embedding_model_name) if uses_emb else 'NO'}")
    print("\n  NOTE: the published dictionary model in 'triage_model/' is left")
    print("        untouched and still loads exactly as before; it is the")
    print("        fallback when sentence-transformers is not installed.")
    print("\nDone.")


if __name__ == "__main__":
    main()
