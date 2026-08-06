"""
stopwords.py
=======================================================================
CONTRIBUTION 1 - Automatic (learned) stop-word removal for Roman Urdu.
=======================================================================

Roman Urdu has no published stop-word list, and hand-writing one
("hai", "hain", "ke", "aur", ...) is subjective and not reproducible.
This module LEARNS the stop-word list from the training complaints
themselves, using two transparent, threshold-based statistics.

A token is a stop word here when BOTH are true:

  1. It is COMMON  - its document frequency (fraction of complaints it
     appears in) is high, i.e. at or above a percentile cut-off of all
     token document frequencies, and at least `min_df`.

  2. It is UNINFORMATIVE - knowing whether the token is present tells
     you almost nothing about the triage level. Measured two ways:
       * normalized mutual information  I(present ; triage) / H(triage),
         which is 0 for a token that is spread evenly over all levels;
       * a chi-square test of independence on the 2 x K contingency
         table, whose p-value must be above `alpha` (i.e. we FAIL to
         reject independence -> no evidence the token carries signal).

Every number behind every decision is written to `learned_stopwords.json`
so the methods section of the paper can quote it and a reviewer can
re-derive the list by hand.

Deliberately NOT a black box: no clustering, no embedding, no learned
threshold. Two statistics and three explicit cut-offs.

-----------------------------------------------------------------------
CLINICAL SAFETY GUARD
-----------------------------------------------------------------------
A purely statistical rule can, on an imbalanced dataset, decide that a
genuine symptom word is "uninformative" simply because it appears in
every triage level (e.g. "dárd" - pain - is present at all four levels).
Deleting a symptom word from a *triage* system is the one failure mode
worth engineering against, so tokens in the project's clinical
vocabulary are protected from removal by default. Every protected token
that the statistics WOULD have dropped is still recorded in the JSON
under "protected_from_removal", so the guard is visible rather than
hidden, and it can be switched off with protect_clinical_terms=False.

-----------------------------------------------------------------------
USAGE
-----------------------------------------------------------------------
    from stopwords import learn_stopwords, remove_stopwords, save_stopwords

    stops, report = learn_stopwords(corpus, labels)
    save_stopwords(report)                       # -> learned_stopwords.json
    clean = remove_stopwords("seena mein dard hai", stops)   # "seena dard"

At inference time the saved list is reloaded with load_stopwords(), so
training and prediction always strip exactly the same tokens.
=======================================================================
"""

import json
import math
import os
from collections import Counter

import numpy as np

# Clinical vocabulary used by the safety guard. Imported from the shared
# pipeline so there is still a single source of truth for medical terms.
from triage_pipeline import (
    CANONICAL_VOCAB,
    DIACRITIZATION_MAP,
    MEDICAL_WEIGHTS,
    make_console_safe,   # re-exported so scripts can import it from here too
)

STOPWORDS_FILE = "learned_stopwords.json"

# Tokens that are statistically empty but semantically loaded: negations
# ("saans NAHI aa rahi" = cannot breathe) and intensity modifiers
# ("TEZ bukhar" = high fever). They are NOT auto-exempted - that would be
# a hidden override of the data - but when the learner selects one it is
# listed under "review_recommended" in the JSON so a clinician signs off.
# TODO(Sir): confirm whether negation/intensity tokens should be exempted
# outright. Doing so needs a negation-aware representation, not just a
# word list, because the embedding step sees bag-of-tokens order anyway.
REVIEW_WATCHLIST = {
    # negation
    "nahi", "nai", "na", "bina", "bagair",
    # intensity / severity modifiers
    "tez", "shadeed", "halka", "halki", "zyada", "kam", "bohat", "thora",
}

# ---- default thresholds (all explicit, all reported in the JSON) ----
DEFAULT_MIN_DF = 0.02          # must appear in >= 2% of complaints
DEFAULT_DF_PERCENTILE = 75.0   # "high frequency" = top 25% of tokens by df
DEFAULT_MI_THRESHOLD = 0.01    # normalized mutual information "near zero"
DEFAULT_ALPHA = 0.05           # chi-square p-value above this = independent


def _protected_terms():
    """Clinical tokens that must never be stripped from a complaint.

    Covers the fuzzy-match vocabulary, every diacritized canonical form
    and its spelling variants, and any attention keyword the project
    boosts (weight > 1.0). Suppressed filler keys such as "hai" (0.6)
    are intentionally NOT protected - those are exactly the words this
    learner is meant to find on its own.
    """
    protected = set(CANONICAL_VOCAB)
    for canonical, variants in DIACRITIZATION_MAP.items():
        protected.add(canonical)
        protected.update(variants)
    protected.update(k for k, w in MEDICAL_WEIGHTS.items() if w > 1.0)
    return protected


def _tokenize(text):
    return str(text).split()


def _contingency(present_flags, labels, classes):
    """2 x K table: row 0 = token absent, row 1 = token present."""
    table = np.zeros((2, len(classes)), dtype=float)
    class_index = {c: i for i, c in enumerate(classes)}
    for present, label in zip(present_flags, labels):
        table[int(present), class_index[label]] += 1
    return table


def _chi_square(table):
    """Chi-square statistic, degrees of freedom and p-value for a 2 x K table."""
    from scipy.stats import chi2 as chi2_dist

    total = table.sum()
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / total

    # Cells with zero expectation contribute nothing and would divide by zero.
    mask = expected > 0
    stat = float(((table[mask] - expected[mask]) ** 2 / expected[mask]).sum())

    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    p_value = float(chi2_dist.sf(stat, dof)) if dof > 0 else 1.0
    return stat, dof, p_value


def _normalized_mutual_information(table):
    """I(present ; label) / H(label), in [0, 1]. 0 = token carries no signal."""
    total = table.sum()
    if total == 0:
        return 0.0

    joint = table / total
    p_present = joint.sum(axis=1, keepdims=True)
    p_label = joint.sum(axis=0, keepdims=True)

    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            pij = joint[i, j]
            if pij > 0 and p_present[i, 0] > 0 and p_label[0, j] > 0:
                mi += pij * math.log2(pij / (p_present[i, 0] * p_label[0, j]))

    label_entropy = -sum(p * math.log2(p) for p in p_label[0] if p > 0)
    if label_entropy <= 0:
        return 0.0
    return float(max(0.0, mi) / label_entropy)


def learn_stopwords(corpus,
                    labels,
                    min_df=DEFAULT_MIN_DF,
                    df_percentile=DEFAULT_DF_PERCENTILE,
                    mi_threshold=DEFAULT_MI_THRESHOLD,
                    alpha=DEFAULT_ALPHA,
                    protect_clinical_terms=True):
    """Learn Roman Urdu stop words from labelled complaints.

    Args:
        corpus: iterable of complaint strings. Pass the SAME text
            representation the stop words will later be removed from
            (in this project: after normalization + fuzzy correction),
            so the learned tokens actually match at inference time.
        labels: triage label per complaint, same length as corpus.
        min_df: minimum document frequency (fraction) to be considered
            common at all.
        df_percentile: a token is "high frequency" when its document
            frequency is at or above this percentile of all token
            document frequencies.
        mi_threshold: normalized mutual information at or below this
            counts as "carries no signal about the triage level".
        alpha: chi-square p-value above this means we cannot reject
            independence between the token and the triage level.
        protect_clinical_terms: keep clinical vocabulary out of the
            stop-word list regardless of the statistics (see module
            docstring).

    Returns:
        (stopwords, report) - a set of tokens, and a dict holding every
        threshold, per-token statistic and decision, ready to be saved
        with save_stopwords() and quoted in the paper.
    """
    corpus = [str(t) for t in corpus]
    labels = list(labels)
    if len(corpus) != len(labels):
        raise ValueError(
            f"corpus and labels must be the same length "
            f"({len(corpus)} vs {len(labels)})")

    n_docs = len(corpus)
    if n_docs == 0:
        raise ValueError("corpus is empty - nothing to learn from")

    doc_tokens = [set(_tokenize(t)) for t in corpus]
    doc_counts = Counter()
    for tokens in doc_tokens:
        doc_counts.update(tokens)

    classes = sorted(set(labels))
    protected = _protected_terms() if protect_clinical_terms else set()

    # --- criterion 1: how common is each token? ---
    doc_freqs = {tok: count / n_docs for tok, count in doc_counts.items()}
    df_cutoff = float(np.percentile(list(doc_freqs.values()), df_percentile))
    high_df_cutoff = max(df_cutoff, min_df)

    # --- criterion 2: does the token discriminate between triage levels? ---
    token_stats = []
    stopwords = set()
    protected_hits = []

    for token in sorted(doc_freqs, key=lambda t: -doc_freqs[t]):
        df = doc_freqs[token]
        is_common = df >= high_df_cutoff

        if not is_common:
            # Rare tokens are never stop words; skip the statistics to keep
            # the report focused on the words that were actually in play.
            continue

        present_flags = [token in tokens for tokens in doc_tokens]
        table = _contingency(present_flags, labels, classes)
        chi2_stat, dof, p_value = _chi_square(table)
        nmi = _normalized_mutual_information(table)

        is_uninformative = (nmi <= mi_threshold) and (p_value > alpha)
        is_protected = token in protected
        is_stopword = is_common and is_uninformative and not is_protected

        if is_stopword:
            stopwords.add(token)
        if is_common and is_uninformative and is_protected:
            protected_hits.append(token)

        token_stats.append({
            "token": token,
            "document_frequency": round(df, 4),
            "document_count": int(doc_counts[token]),
            "normalized_mutual_information": round(nmi, 6),
            "chi_square": round(chi2_stat, 4),
            "chi_square_dof": dof,
            "chi_square_p_value": round(p_value, 6),
            "is_high_frequency": bool(is_common),
            "is_uninformative": bool(is_uninformative),
            "clinically_protected": bool(is_protected),
            "is_stopword": bool(is_stopword),
        })

    report = {
        "method": (
            "A token is a stop word when its document frequency is at or above "
            "the df cut-off AND its normalized mutual information with the "
            "triage label is at or below mi_threshold AND its chi-square test "
            "of independence is not significant (p > alpha). Clinical "
            "vocabulary is exempt when protect_clinical_terms is true."
        ),
        "thresholds": {
            "min_df": min_df,
            "df_percentile": df_percentile,
            "df_cutoff_from_percentile": round(df_cutoff, 6),
            "effective_df_cutoff": round(high_df_cutoff, 6),
            "mi_threshold": mi_threshold,
            "alpha": alpha,
            "protect_clinical_terms": bool(protect_clinical_terms),
        },
        "corpus": {
            "n_documents": n_docs,
            "n_unique_tokens": len(doc_counts),
            "n_tokens_tested": len(token_stats),
            "triage_classes": [int(c) if isinstance(c, (int, np.integer)) else str(c)
                               for c in classes],
        },
        "n_stopwords": len(stopwords),
        "stopwords": sorted(stopwords),
        "protected_from_removal": sorted(protected_hits),
        "review_recommended": sorted(stopwords & REVIEW_WATCHLIST),
        "review_recommended_note": (
            "These tokens were learned as stop words because they are "
            "statistically uninformative on this dataset, but they are "
            "negations or intensity modifiers and carry clinical meaning. "
            "A clinician should confirm before they are stripped."
        ),
        "token_statistics": token_stats,
    }

    return stopwords, report


def remove_stopwords(text, stopword_set):
    """Drop every learned stop word from a complaint string.

    Never returns an empty string: if a complaint is made entirely of
    stop words the original text is kept, because an empty complaint
    would embed to a meaningless vector and silently lose the patient's
    only free-text signal.
    """
    if not stopword_set:
        return str(text)
    kept = [w for w in _tokenize(text) if w not in stopword_set]
    if not kept:
        return str(text)
    return " ".join(kept)


def save_stopwords(report, path=STOPWORDS_FILE):
    """Write the full learned-stop-word report (list + statistics) to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


def load_stopwords(path=STOPWORDS_FILE):
    """Load the learned stop-word set. Returns an empty set if not learned yet.

    Returning empty (rather than raising) keeps every existing predictor
    working on a fresh clone where training has not been run.
    """
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return set(report.get("stopwords", []))


def summarize(report, top_n=15):
    """Human-readable summary for the training console output."""
    t = report["thresholds"]
    c = report["corpus"]
    lines = [
        "AUTOMATIC STOP-WORD LEARNER  (Contribution 1)",
        f"  complaints analysed        : {c['n_documents']}",
        f"  unique tokens              : {c['n_unique_tokens']}",
        f"  high-frequency tokens sifted: {c['n_tokens_tested']} "
        f"(document frequency >= {t['effective_df_cutoff']:.4f})",
        f"  rule: df >= {t['effective_df_cutoff']:.4f}  AND  "
        f"normalized MI <= {t['mi_threshold']}  AND  chi-square p > {t['alpha']}",
        f"  stop words learned         : {report['n_stopwords']}",
    ]
    stops = report["stopwords"]
    if stops:
        shown = ", ".join(stops[:top_n])
        more = "" if len(stops) <= top_n else f", ... (+{len(stops) - top_n} more)"
        lines.append(f"  learned list               : {shown}{more}")
    if report["protected_from_removal"]:
        lines.append(
            "  kept despite the statistics (clinical safety guard): "
            + ", ".join(report["protected_from_removal"]))
    if report.get("review_recommended"):
        lines.append(
            "  [!] needs clinician sign-off (negation / intensity words): "
            + ", ".join(report["review_recommended"]))
    return "\n".join(lines)


# ======================================================================
# Standalone run:  python stopwords.py
# Learns the list from the dataset and writes learned_stopwords.json.
# The training script calls learn_stopwords() directly, so running this
# by hand is only needed to regenerate the list for the paper.
# ======================================================================

def _main():
    import argparse
    import pandas as pd
    from triage_pipeline import normalize_roman_urdu

    parser = argparse.ArgumentParser(
        description="Learn Roman Urdu stop words from the triage dataset.")
    parser.add_argument("--data", default="triage_mixed_language_dataset.csv")
    parser.add_argument("--out", default=STOPWORDS_FILE)
    parser.add_argument("--min-df", type=float, default=DEFAULT_MIN_DF)
    parser.add_argument("--df-percentile", type=float, default=DEFAULT_DF_PERCENTILE)
    parser.add_argument("--mi-threshold", type=float, default=DEFAULT_MI_THRESHOLD)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--no-clinical-guard", action="store_true",
                        help="allow clinical vocabulary to be removed too")
    args = parser.parse_args()

    make_console_safe()

    df = pd.read_csv(args.data)
    print(f"[ok] Loaded {len(df)} complaints from {args.data}")

    # Learn on the same representation the stop words are applied to:
    # normalized + fuzzy-corrected text (see ARCHITECTURE.md, Task 2).
    corpus = df["Complaint_Text"].fillna("").astype(str).apply(normalize_roman_urdu)
    labels = df["Triage_Level"].tolist()

    stops, report = learn_stopwords(
        corpus, labels,
        min_df=args.min_df,
        df_percentile=args.df_percentile,
        mi_threshold=args.mi_threshold,
        alpha=args.alpha,
        protect_clinical_terms=not args.no_clinical_guard,
    )

    print()
    print(summarize(report, top_n=50))
    path = save_stopwords(report, args.out)
    print(f"\n[ok] Full report ({len(report['token_statistics'])} tokens with "
          f"per-token statistics) saved to '{path}'")

    examples = [
        "seena mein shadeed dard aur pasina aa raha hai",
        "subah se pait mein tez dard hai lekin bukhar nahi",
    ]
    print("\nExamples (raw -> normalized -> stop words removed):")
    for example in examples:
        normalized = normalize_roman_urdu(example)
        print(f"  raw        : {example}")
        print(f"  normalized : {normalized}")
        print(f"  cleaned    : {remove_stopwords(normalized, stops)}")
        print()


if __name__ == "__main__":
    _main()
