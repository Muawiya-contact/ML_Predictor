"""
embedding_evaluation.py
=======================================================================
CONTRIBUTION 2 - How effective is the embedding generator?
=======================================================================

This script does NOT touch the classifier. It measures how faithfully
the sentence-embedding model represents Roman Urdu ED complaints, which
is the second contribution the paper needs.

Three measurements, per ARCHITECTURE.md Task 4:

  4b. WITHIN-CLUSTER PAIRWISE SIMILARITY
      For each manual meaning-cluster, embed every complaint and compare
      every possible PAIR. A cluster of 10 complaints gives exactly
      C(10,2) = 10! / (2! * 8!) = 45 pairs. For each pair we report
        * cosine similarity  (0-1 after clipping; 1 = identical meaning)
        * Pearson correlation between the two embedding vectors
        * Euclidean distance (0 = identical; lower = more similar)
      and the fraction of pairs above the 0.5 similarity threshold -
      Sir's cut-off for "these two complaints are correctly seen as
      similar".

  4b(ii). ACROSS-CLUSTER SIMILARITY
      A good embedding generator gives HIGH within-cluster similarity
      and LOWER across-cluster similarity. Within-cluster similarity
      alone proves nothing: a model that maps every sentence to the same
      vector would score 1.00. The separation gap is the real evidence.

  4c. ROUND-TRIP FIDELITY  (nearest-neighbour method)
      Sir's description is "convert text to embedding, then convert the
      embedding back to text and check it is the same". Standard
      sentence embeddings are ONE-WAY and lossy - the vector simply does
      not contain the information needed to reconstruct the sentence, so
      literal inversion is impossible without special (English-only)
      inversion models. See ARCHITECTURE.md Task 4c.

      The implementable test that measures the same property:
        1. embed every complaint,
        2. for each one find the closest OTHER embedding,
        3. check whether that nearest neighbour means the same thing
           (i.e. belongs to the same manual cluster),
        4. EFFICIENCY SCORE = fraction of complaints whose nearest
           neighbour is a correct same-meaning complaint.

      Reported twice:
        * CLOSED POOL - neighbours drawn only from the clustered
          complaints. This is the number the design document asks for.
        * OPEN POOL   - neighbours drawn from the clustered complaints
          PLUS every other complaint in the dataset acting as a
          distractor. Harder and closer to real deployment, where a new
          complaint is compared against the whole record set.

-----------------------------------------------------------------------
HOW TO RUN
-----------------------------------------------------------------------
    pip install -r requirements-embedding.txt      # once, needs internet
    python embedding_evaluation.py

    python embedding_evaluation.py --raw            # skip preprocessing
    python embedding_evaluation.py --threshold 0.6
    python embedding_evaluation.py --model <other-sentence-transformer>

Outputs:
    embedding_evaluation_results.csv        per-cluster summary
    embedding_evaluation_pairs.csv          EVERY pair (45 per 10-cluster)
    embedding_evaluation_neighbours.csv     nearest neighbour of each complaint
=======================================================================
"""

import argparse
import itertools
import json
import os
import sys
import warnings

# See prediction.py: keeps `import triage_pipeline` working regardless of
# how this script was launched.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

from triage_pipeline import (make_console_safe,
                             preprocess_corpus_for_embedding, project_path,
                             resolve_project_file)

warnings.filterwarnings("ignore")
make_console_safe()

CLUSTERS_FILE = "evaluation_clusters.json"
DATA_FILE = "triage_mixed_language_dataset.csv"
# Outputs go beside the code, so the GUI's Results tabs find them however
# this study was launched. Inputs are resolved the same way when read.
SUMMARY_CSV = project_path("embedding_evaluation_results.csv")
PAIRS_CSV = project_path("embedding_evaluation_pairs.csv")
NEIGHBOURS_CSV = project_path("embedding_evaluation_neighbours.csv")

# Same default as the training pipeline, so the study evaluates the model
# the system actually uses. TODO(Sir): standardize (ARCHITECTURE.md s6 Q1).
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_THRESHOLD = 0.5      # Sir's "correctly seen as similar" cut-off


# ----------------------------------------------------------------------
# Pairwise measures
# ----------------------------------------------------------------------

def cosine_similarity(a, b):
    """Cosine similarity, clipped to 0-1 as the design document expects.

    Raw cosine runs -1..1. Negative values mean 'pointing away', which for
    this report is simply 'not similar', so the range is clipped at 0
    rather than rescaled - rescaling would turn 'unrelated' (0.0) into a
    misleading 0.5.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def pearson_correlation(a, b):
    """Pearson correlation between two embedding vectors ('correlation').

    The design document asks for "correlation"; cosine similarity is the
    standard measure for embeddings, so both are reported. Expect them to
    agree almost exactly: Pearson is the cosine of the MEAN-CENTRED
    vectors, and transformer embedding vectors already have a mean very
    close to zero. In this study they differ only where cosine was
    clipped at 0 (i.e. the pair was genuinely anti-correlated). That is
    not a bug - it is a property of the representation, and worth one
    line in the paper so a reviewer is not puzzled by two near-identical
    columns.
    """
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))


def n_pairs(n):
    """C(n,2) = n! / (2! (n-2)!) - the count the design document quotes."""
    return n * (n - 1) // 2


# ----------------------------------------------------------------------

def load_embedding_model(model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n[x] 'sentence-transformers' is not installed.")
        print("    Install it once (needs internet), then re-run:")
        print("        pip install -r requirements-embedding.txt\n")
        sys.exit(1)

    print(f"Loading embedding model: {model_name}")
    print("(first run downloads it; later runs are fully offline)")
    try:
        return SentenceTransformer(model_name, device="cpu")
    except Exception as e:
        print("\n[x] Could not load the embedding model. The first run needs")
        print(f"    internet to download it once. Underlying error:\n    {e}\n")
        sys.exit(1)


def load_clusters(path):
    path = resolve_project_file(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    clusters = data["clusters"]

    # A complaint in two clusters would make the nearest-neighbour test
    # ill-defined, so fail loudly rather than silently score nonsense.
    seen = {}
    for name, items in clusters.items():
        for text in items:
            if text in seen:
                raise ValueError(
                    f"Complaint appears in two clusters ('{seen[text]}' and "
                    f"'{name}'), which breaks the fidelity test:\n  {text}")
            seen[text] = name
    return clusters


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measure how faithfully the embedding model represents complaints.")
    parser.add_argument("--clusters", default=CLUSTERS_FILE)
    parser.add_argument("--data", default=DATA_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="similarity cut-off for 'correctly seen as similar'")
    parser.add_argument("--raw", action="store_true",
                        help="embed raw text instead of preprocessed text")
    args = parser.parse_args()

    print("=" * 78)
    print("EMBEDDING GENERATOR EFFECTIVENESS STUDY  (Contribution 2)")
    print("=" * 78)

    clusters = load_clusters(args.clusters)
    names = list(clusters)
    texts, labels = [], []
    for name in names:
        for t in clusters[name]:
            texts.append(t)
            labels.append(name)

    print(f"[ok] {len(names)} manual clusters, {len(texts)} complaints "
          f"({args.clusters})")
    for name in names:
        n = len(clusters[name])
        print(f"       {name:<24} {n:>3} complaints -> {n_pairs(n):>3} pairs")
    print(f"[ok] total pairs to be reported: "
          f"{sum(n_pairs(len(clusters[n])) for n in names)}")

    # ---- preprocessing (same chain the classifier uses) ----
    if args.raw:
        model_input = list(texts)
        print("\n[!] --raw: embedding the raw text (no fuzzy / stop-word step)")
    else:
        model_input = preprocess_corpus_for_embedding(texts)
        print("\n[ok] Preprocessing applied: clean -> fuzzy -> learned stop words")

    emb_model = load_embedding_model(args.model)
    vectors = emb_model.encode(model_input, batch_size=32, convert_to_numpy=True,
                               show_progress_bar=False, normalize_embeddings=False)
    print(f"[ok] Embedded {vectors.shape[0]} complaints x {vectors.shape[1]} dims")

    # ==================================================================
    # 4b - within-cluster pairwise similarity
    # ==================================================================
    print("\n" + "=" * 78)
    print("PART 4b  -  WITHIN-CLUSTER PAIRWISE SIMILARITY")
    print("=" * 78)
    print(f"Threshold for 'correctly seen as similar': {args.threshold}\n")

    index_of = {name: [i for i, l in enumerate(labels) if l == name] for name in names}
    pair_rows, summary_rows = [], []

    for name in names:
        idx = index_of[name]
        sims, dists, corrs = [], [], []
        for a, b in itertools.combinations(idx, 2):
            s = cosine_similarity(vectors[a], vectors[b])
            d = euclidean_distance(vectors[a], vectors[b])
            r = pearson_correlation(vectors[a], vectors[b])
            sims.append(s); dists.append(d); corrs.append(r)
            pair_rows.append({
                "cluster": name,
                "complaint_a": texts[a],
                "complaint_b": texts[b],
                "cosine_similarity": round(s, 4),
                "pearson_correlation": round(r, 4),
                "euclidean_distance": round(d, 4),
                "above_threshold": bool(s > args.threshold),
            })

        sims = np.array(sims)
        above = int((sims > args.threshold).sum())
        pass_rate = 100 * above / len(sims) if len(sims) else 0.0
        summary_rows.append({
            "cluster": name,
            "n_complaints": len(idx),
            "n_pairs": len(sims),
            "mean_cosine_similarity": round(float(sims.mean()), 4),
            "min_cosine_similarity": round(float(sims.min()), 4),
            "max_cosine_similarity": round(float(sims.max()), 4),
            "mean_pearson_correlation": round(float(np.mean(corrs)), 4),
            "mean_euclidean_distance": round(float(np.mean(dists)), 4),
            f"pairs_above_{args.threshold}": above,
            f"pct_above_{args.threshold}": round(pass_rate, 1),
        })

        flag = "ok " if pass_rate >= 80 else ("weak" if pass_rate >= 50 else "POOR")
        print(f"  [{flag}] {name:<22} pairs={len(sims):>3}  "
              f"mean_sim={sims.mean():.3f}  mean_dist={np.mean(dists):.3f}  "
              f"above {args.threshold}: {above}/{len(sims)} ({pass_rate:.1f}%)")

    # ==================================================================
    # 4b(ii) - across-cluster separation
    # ==================================================================
    within_all, across_all = [], []
    for a, b in itertools.combinations(range(len(texts)), 2):
        s = cosine_similarity(vectors[a], vectors[b])
        (within_all if labels[a] == labels[b] else across_all).append(s)
    within_mean, across_mean = float(np.mean(within_all)), float(np.mean(across_all))
    gap = within_mean - across_mean

    print("\n" + "-" * 78)
    print("SEPARATION  (within-cluster similarity must beat across-cluster)")
    print("-" * 78)
    print(f"  mean WITHIN-cluster  similarity : {within_mean:.3f}   (want HIGH)")
    print(f"  mean ACROSS-cluster  similarity : {across_mean:.3f}   (want LOW)")
    print(f"  separation gap                  : {gap:+.3f}", end="   ")
    print("-> meanings separate well" if gap > 0.15 else
          "-> WEAK separation; Roman Urdu may not cluster well with this model")

    # ==================================================================
    # 4c - nearest-neighbour round-trip fidelity
    # ==================================================================
    print("\n" + "=" * 78)
    print("PART 4c  -  ROUND-TRIP FIDELITY  (nearest-neighbour method)")
    print("=" * 78)
    print("Literal embedding inversion is not possible - sentence embeddings")
    print("are lossy and one-way. Instead: for each complaint, find its closest")
    print("neighbour and check whether it means the same thing.\n")

    # Cosine similarity matrix over the clustered pool.
    norms = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    sim_matrix = norms @ norms.T
    np.fill_diagonal(sim_matrix, -np.inf)          # never match yourself
    nn_closed = np.argmax(sim_matrix, axis=1)

    neighbour_rows = []
    correct_closed = 0
    for i, j in enumerate(nn_closed):
        hit = labels[i] == labels[j]
        correct_closed += hit
        neighbour_rows.append({
            "complaint": texts[i],
            "true_cluster": labels[i],
            "nearest_neighbour": texts[j],
            "neighbour_cluster": labels[j],
            "similarity": round(float(sim_matrix[i, j]), 4),
            "same_meaning": bool(hit),
        })
    closed_score = 100 * correct_closed / len(texts)

    # ---- open pool: every other dataset complaint as a distractor ----
    open_score = None
    try:
        df = pd.read_csv(resolve_project_file(args.data))
        pool = [t for t in df["Complaint_Text"].dropna().astype(str).unique()
                if t not in set(texts)]
    except FileNotFoundError:
        pool = []

    if pool:
        pool_input = list(pool) if args.raw else preprocess_corpus_for_embedding(pool)
        pool_vecs = emb_model.encode(pool_input, batch_size=32, convert_to_numpy=True,
                                     show_progress_bar=False, normalize_embeddings=False)
        pool_norms = pool_vecs / np.clip(
            np.linalg.norm(pool_vecs, axis=1, keepdims=True), 1e-12, None)
        cross = norms @ pool_norms.T          # clustered x distractors

        correct_open = 0
        for i in range(len(texts)):
            best_in = float(sim_matrix[i].max())
            best_out = float(cross[i].max())
            if best_in >= best_out:
                correct_open += labels[i] == labels[nn_closed[i]]
                neighbour_rows[i]["open_pool_neighbour"] = texts[nn_closed[i]]
            else:
                neighbour_rows[i]["open_pool_neighbour"] = pool[int(np.argmax(cross[i]))]
            neighbour_rows[i]["open_pool_same_meaning"] = bool(best_in >= best_out and
                                                              labels[i] == labels[nn_closed[i]])
        open_score = 100 * correct_open / len(texts)

    print(f"  EMBEDDING GENERATOR EFFICIENCY (closed pool) : {closed_score:.1f}%")
    print(f"    {correct_closed} of {len(texts)} complaints had a same-meaning "
          f"nearest neighbour.")
    if open_score is not None:
        print(f"  EMBEDDING GENERATOR EFFICIENCY (open pool)   : {open_score:.1f}%")
        print(f"    with all {len(pool)} other dataset complaints as distractors.")

    print("\n  Per-cluster breakdown (closed pool):")
    for name in names:
        idx = index_of[name]
        hits = sum(1 for i in idx if labels[nn_closed[i]] == name)
        pct = 100 * hits / len(idx)
        flag = "ok " if pct >= 80 else ("weak" if pct >= 50 else "POOR")
        print(f"    [{flag}] {name:<22} {hits}/{len(idx)}  ({pct:.0f}%)")

    worst = [r for r in neighbour_rows if not r["same_meaning"]]
    if worst:
        print(f"\n  Complaints whose nearest neighbour was a DIFFERENT meaning "
              f"({len(worst)}):")
        for r in worst[:10]:
            print(f"    '{r['complaint']}'")
            print(f"        -> '{r['nearest_neighbour']}'  "
                  f"[{r['neighbour_cluster']}, sim={r['similarity']:.3f}]")
        if len(worst) > 10:
            print(f"    ... and {len(worst) - 10} more (see {NEIGHBOURS_CSV})")

    # ==================================================================
    # Save everything for the article
    # ==================================================================
    summary = pd.DataFrame(summary_rows)
    overall = {
        "cluster": "OVERALL",
        "n_complaints": len(texts),
        "n_pairs": int(summary["n_pairs"].sum()),
        "mean_cosine_similarity": round(within_mean, 4),
        "min_cosine_similarity": round(float(summary["min_cosine_similarity"].min()), 4),
        "max_cosine_similarity": round(float(summary["max_cosine_similarity"].max()), 4),
        "mean_pearson_correlation": round(float(summary["mean_pearson_correlation"].mean()), 4),
        "mean_euclidean_distance": round(float(summary["mean_euclidean_distance"].mean()), 4),
        f"pairs_above_{args.threshold}": int(summary[f"pairs_above_{args.threshold}"].sum()),
        f"pct_above_{args.threshold}": round(
            100 * summary[f"pairs_above_{args.threshold}"].sum()
            / summary["n_pairs"].sum(), 1),
        "across_cluster_similarity": round(across_mean, 4),
        "separation_gap": round(gap, 4),
        "nn_efficiency_closed_pool_pct": round(closed_score, 1),
        "nn_efficiency_open_pool_pct": (None if open_score is None
                                        else round(open_score, 1)),
        "embedding_model": args.model,
        "embedding_dim": int(vectors.shape[1]),
        "preprocessing": "raw" if args.raw else "fuzzy + learned stop-word removal",
    }
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)

    summary.to_csv(SUMMARY_CSV, index=False)
    pd.DataFrame(pair_rows).to_csv(PAIRS_CSV, index=False)
    pd.DataFrame(neighbour_rows).to_csv(NEIGHBOURS_CSV, index=False)

    print("\n" + "=" * 78)
    print("SAVED")
    print("=" * 78)
    print(f"  {SUMMARY_CSV:<36} per-cluster summary + overall row")
    print(f"  {PAIRS_CSV:<36} every pair ({len(pair_rows)} rows)")
    print(f"  {NEIGHBOURS_CSV:<36} nearest neighbour of each complaint")

    print("\n" + "=" * 78)
    print("HEADLINE NUMBERS FOR THE PAPER")
    print("=" * 78)
    print(f"  Embedding model                    : {args.model}")
    print(f"  Embedding dimension                : {vectors.shape[1]}")
    print(f"  Mean within-cluster similarity     : {within_mean:.3f}")
    print(f"  Mean across-cluster similarity     : {across_mean:.3f}")
    print(f"  Separation gap                     : {gap:+.3f}")
    print(f"  Pairs above {args.threshold} threshold          : "
          f"{overall[f'pairs_above_{args.threshold}']}/{overall['n_pairs']} "
          f"({overall[f'pct_above_{args.threshold}']}%)")
    print(f"  Embedding generator efficiency     : {closed_score:.1f}% (closed pool)")
    if open_score is not None:
        print(f"                                       {open_score:.1f}% (open pool)")
    print("\nDone.")


if __name__ == "__main__":
    main()
