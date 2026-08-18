"""
embedding_evaluation_translation_compare.py
=======================================================================
Does translating to English beat normalizing the Roman Urdu?
=======================================================================

Step 4 of the translation experiment. Runs the SAME within-cluster /
across-cluster measurement embedding_evaluation.py performs, twice, on
the same complaints:

  (a) ROMAN URDU  - the deployed pipeline, unchanged:
                    clean -> rule replace -> fuzzy -> diacritize ->
                    learned stop-word removal -> sentence-transformer

  (b) ENGLISH     - the gpt-4o-mini translation of the same complaint,
                    fed straight to the same sentence-transformer with
                    NO Roman Urdu normalization.

WHY (b) SKIPS NORMALIZATION
---------------------------
Because running English through the Roman Urdu pipeline does not leave
it in English. The dictionary is effectively bidirectional: "Severe
chest pain radiating to the left arm" comes back as "shadīd sēna dárd
phailna to the bāyāñ bāzū". Normalizing the English arm would translate
it back into Roman Urdu canonicals and the comparison would be measuring
the dictionary twice rather than comparing two languages.

WHAT THE NUMBERS MEAN
---------------------
within-cluster   mean cosine similarity between complaints that mean the
                 same thing. Higher is better.
across-cluster   mean cosine similarity between complaints from
                 DIFFERENT meaning groups. Lower is better.
separation gap   within minus across. This is the number that matters:
                 a model can score high within-cluster simply by mapping
                 every cardiac complaint to nearly the same vector, and
                 the gap is what exposes that.
pass rate        fraction of within-cluster pairs above --threshold.

USAGE
-----
    python embedding_evaluation_translation_compare.py
    python embedding_evaluation_translation_compare.py --threshold 0.6
=======================================================================
"""

import argparse
import itertools
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

DEFAULT_DATA = "cardiac_multilingual_10000_v3_english.csv"
DEFAULT_THRESHOLD = 0.5
PER_CLUSTER = 10
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

#: Meaning groups, detected from the CANONICAL tokens the Roman Urdu
#: pipeline produces. Grouping on canonical form rather than on raw
#: spelling means membership is decided identically for both arms - the
#: English arm is scored on exactly the rows the Roman Urdu arm selected,
#: so the two columns are comparable rather than two different samples.
CLUSTER_RULES = {
    "chest_pain":      lambda s: "sēna" in s and "dárd" in s,
    "breathlessness":  lambda s: "sāns" in s or "phūlna" in s,
    "palpitations":    lambda s: "dháḍkan" in s,
    "syncope":         lambda s: "bēhōsh" in s,
    "chest_burning":   lambda s: "jálan" in s,
    "chest_tightness": lambda s: "jákṛan" in s or "dabāo" in s,
}


def build_clusters(df, seed):
    """Pick PER_CLUSTER complaints per meaning group, from BOTH columns."""
    import triage_pipeline as tp

    norm = df["Complaint_Text"].astype(str).map(tp.normalize_roman_urdu)
    clusters = {}
    used = set()
    for name, matches in CLUSTER_RULES.items():
        idx = [i for i, s in zip(df.index, norm)
               if matches(s) and i not in used
               and isinstance(df.at[i, "English_Translation"], str)
               and df.at[i, "English_Translation"].strip()]
        if len(idx) < PER_CLUSTER:
            print(f"  [skip] {name}: only {len(idx)} usable rows")
            continue
        chosen = list(pd.Series(idx).sample(PER_CLUSTER, random_state=seed))
        used.update(chosen)          # a complaint belongs to ONE cluster
        clusters[name] = chosen
    return clusters


def measure(enc, texts_by_cluster, threshold):
    """Within/across similarity, separation gap and pass rate."""
    vecs = {k: enc.encode(v, convert_to_numpy=True, normalize_embeddings=True,
                          show_progress_bar=False)
            for k, v in texts_by_cluster.items()}
    rows = {}
    for k, v in vecs.items():
        sims = [float(np.dot(v[i], v[j]))
                for i, j in itertools.combinations(range(len(v)), 2)]
        others = [float(np.dot(a, b))
                  for k2, v2 in vecs.items() if k2 != k
                  for a in v for b in v2]
        rows[k] = {
            "within": float(np.mean(sims)),
            "across": float(np.mean(others)),
            "gap": float(np.mean(sims)) - float(np.mean(others)),
            "pass_rate": 100.0 * sum(s > threshold for s in sims) / len(sims),
            "n_pairs": len(sims),
        }
    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    import triage_pipeline as tp
    from stopwords import load_stopwords, remove_stopwords

    tp.make_console_safe()
    df = pd.read_csv(args.data)
    if "English_Translation" not in df.columns:
        raise SystemExit(f"{args.data} has no English_Translation column. "
                         f"Run translate_roman_urdu.translate_dataset_column first.")

    n_translated = df["English_Translation"].notna().sum()
    print(f"[ok] {args.data}: {len(df)} rows, {n_translated} translated")
    if n_translated < len(df):
        print(f"[warn] {len(df) - n_translated} rows have no translation and "
              f"are excluded from cluster selection")

    enc = tp.load_sentence_transformer(EMBEDDING_MODEL)
    stops = load_stopwords()

    print("\nBuilding meaning clusters from canonical tokens...")
    clusters = build_clusters(df, args.seed)
    if not clusters:
        raise SystemExit("No cluster reached the minimum size.")

    urdu = {k: [remove_stopwords(
                    tp.normalize_roman_urdu(str(df.at[i, "Complaint_Text"])), stops)
                for i in idx]
            for k, idx in clusters.items()}
    english = {k: [str(df.at[i, "English_Translation"]) for i in idx]
               for k, idx in clusters.items()}

    print("Encoding (a) Roman Urdu, normalized...")
    a = measure(enc, urdu, args.threshold)
    print("Encoding (b) English translation, no normalization...")
    b = measure(enc, english, args.threshold)

    print("\n" + "=" * 100)
    print("ROMAN URDU (normalized)  vs  ENGLISH (translated, no normalization)")
    print("=" * 100)
    head = (f"{'meaning group':18s}"
            f"{'within a':>10s}{'within b':>10s}"
            f"{'across a':>10s}{'across b':>10s}"
            f"{'gap a':>9s}{'gap b':>9s}"
            f"{'pass a':>9s}{'pass b':>9s}")
    print(head)
    print("-" * len(head))
    for k in clusters:
        ra, rb = a[k], b[k]
        print(f"{k:18s}"
              f"{ra['within']:10.3f}{rb['within']:10.3f}"
              f"{ra['across']:10.3f}{rb['across']:10.3f}"
              f"{ra['gap']:9.3f}{rb['gap']:9.3f}"
              f"{ra['pass_rate']:8.1f}%{rb['pass_rate']:8.1f}%")
    print("-" * len(head))
    mean = lambda r, f: float(np.mean([r[k][f] for k in clusters]))
    print(f"{'MEAN':18s}"
          f"{mean(a,'within'):10.3f}{mean(b,'within'):10.3f}"
          f"{mean(a,'across'):10.3f}{mean(b,'across'):10.3f}"
          f"{mean(a,'gap'):9.3f}{mean(b,'gap'):9.3f}"
          f"{mean(a,'pass_rate'):8.1f}%{mean(b,'pass_rate'):8.1f}%")
    print("=" * 100)
    print("a = Roman Urdu through the deployed pipeline")
    print("b = gpt-4o-mini English translation, straight into the same encoder")
    print("\nSEPARATION GAP is the column to read. Within-cluster similarity")
    print("alone can be raised by mapping every cardiac complaint to nearly the")
    print("same vector, which helps nobody; the gap subtracts that effect out.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"threshold": args.threshold, "clusters": list(clusters),
                       "roman_urdu": a, "english": b}, f, indent=2)
        print(f"\n[ok] JSON written to {args.json_out}")


if __name__ == "__main__":
    main()
