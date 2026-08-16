"""
check_embedding_pairs.py
=======================================================================
Verify the "same meaning -> same embedding" claim in SUBMISSION_SUMMARY.md.
=======================================================================

WHAT THIS MEASURES
------------------
Five complaint pairs. Each pair says the same clinical thing twice - once
in Roman Urdu, once in English. A nurse typing either one should land in
the same place, so the two should embed close together.

For every pair the script reports cosine similarity twice:

  RAW        the two strings fed straight to the sentence-transformer.
             This is what the encoder does on its own, with no help.
  NORMALIZED the same two strings after the project's text pipeline
             (clean -> rule replace -> fuzzy -> diacritize -> learned
             stop-word removal). This is what the deployed model sees.

The gap between those two columns IS Contribution 1. If normalization
were doing nothing, the columns would match.

WHY IT IS A SCRIPT AND NOT A NOTEBOOK CELL
------------------------------------------
The numbers quoted in SUBMISSION_SUMMARY.md have to be reproducible by
anyone reviewing the submission, including on a machine with no network
(the encoder is loaded from the local cache, same as the deployed model).
Running this prints the current numbers next to the documented ones and
says PASS or MISMATCH per pair, so drift between the code and the write-up
cannot go unnoticed.

USAGE
-----
    python check_embedding_pairs.py
    python check_embedding_pairs.py --threshold 0.6
    python check_embedding_pairs.py --json          # machine-readable
=======================================================================
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

#: The five pairs quoted in SUBMISSION_SUMMARY.md, with the similarity
#: recorded there at the time of writing. `documented` is what the summary
#: claims; the script recomputes it and flags any drift.
PAIRS = [
    {"urdu": "seena mein dard",      "english": "chest pain",
     "concept": "chest pain",        "documented": 0.797},
    {"urdu": "dil ki dhadkan tez",   "english": "heart racing fast",
     "concept": "palpitations",      "documented": 0.824},
    {"urdu": "saans phool rahi hai", "english": "shortness of breath",
     "concept": "breathlessness",    "documented": 0.581},
    {"urdu": "behoshi ho gayi thi",  "english": "patient fainted",
     "concept": "syncope",           "documented": 0.765},
    {"urdu": "seena mein jalan",     "english": "burning sensation in chest",
     "concept": "chest burning",     "documented": 0.640},
]

DOCUMENTED_MEAN_RAW = 0.159
DOCUMENTED_MEAN_NORM = 0.721
DEFAULT_THRESHOLD = 0.5

#: How far a recomputed similarity may drift from the documented figure
#: before it is called a mismatch. Encoding is deterministic on a fixed
#: model, so anything beyond rounding means something actually changed.
DRIFT_TOLERANCE = 0.02


def cosine(a, b):
    return float(np.dot(a, b))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="similarity at or below which a pair is a FAIL "
                        f"(default {DEFAULT_THRESHOLD})")
    p.add_argument("--json", action="store_true",
                   help="emit results as JSON instead of a table")
    args = p.parse_args()

    import triage_pipeline as tp
    from stopwords import load_stopwords, remove_stopwords

    tp.make_console_safe()
    stops = load_stopwords()
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    enc = tp.load_sentence_transformer(model_name)

    def normalize(text):
        return remove_stopwords(tp.normalize_roman_urdu(text), stops)

    results = []
    for pair in PAIRS:
        u, e = pair["urdu"], pair["english"]
        vu, ve = enc.encode([u, e], convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        raw = cosine(vu, ve)

        nu, ne = normalize(u), normalize(e)
        wu, we = enc.encode([nu, ne], convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        norm = cosine(wu, we)

        results.append({
            "concept": pair["concept"],
            "urdu": u, "english": e,
            "urdu_normalized": nu, "english_normalized": ne,
            "raw": round(raw, 4),
            "normalized": round(norm, 4),
            "lift": round(norm - raw, 4),
            "documented": pair["documented"],
            "drift": round(norm - pair["documented"], 4),
            "passes_threshold": bool(norm > args.threshold),
            "matches_documented": bool(abs(norm - pair["documented"]) <= DRIFT_TOLERANCE),
        })

    mean_raw = float(np.mean([r["raw"] for r in results]))
    mean_norm = float(np.mean([r["normalized"] for r in results]))
    failed = [r for r in results if not r["passes_threshold"]]
    drifted = [r for r in results if not r["matches_documented"]]

    if args.json:
        print(json.dumps({
            "model": model_name,
            "threshold": args.threshold,
            "pairs": results,
            "mean_raw": round(mean_raw, 4),
            "mean_normalized": round(mean_norm, 4),
            "mean_lift": round(mean_norm - mean_raw, 4),
            "documented_mean_raw": DOCUMENTED_MEAN_RAW,
            "documented_mean_normalized": DOCUMENTED_MEAN_NORM,
            "pairs_below_threshold": len(failed),
            "pairs_drifted_from_docs": len(drifted),
        }, indent=2, ensure_ascii=False))
        return 0 if not failed and not drifted else 1

    print("=" * 78)
    print("SAME-MEANING EMBEDDING CHECK   (SUBMISSION_SUMMARY.md verification)")
    print("=" * 78)
    print(f"encoder   : {model_name}")
    print(f"stop words: {len(stops)} learned")
    print(f"threshold : {args.threshold}\n")

    for r in results:
        mark = "PASS" if r["passes_threshold"] else "FAIL"
        doc = "ok" if r["matches_documented"] else f"MISMATCH (doc {r['documented']})"
        print(f"[{mark}] {r['concept']}")
        print(f"       urdu     : {r['urdu']!r}")
        print(f"                  -> {r['urdu_normalized']!r}")
        print(f"       english  : {r['english']!r}")
        print(f"                  -> {r['english_normalized']!r}")
        print(f"       raw {r['raw']:.3f}  ->  normalized {r['normalized']:.3f}"
              f"   (lift {r['lift']:+.3f})   vs docs: {doc}")
        print()

    print("-" * 78)
    print(f"  mean raw        : {mean_raw:.3f}   (documented {DOCUMENTED_MEAN_RAW})")
    print(f"  mean normalized : {mean_norm:.3f}   (documented {DOCUMENTED_MEAN_NORM})")
    print(f"  mean lift       : {mean_norm - mean_raw:+.3f}")
    print(f"  pairs above {args.threshold}: {len(results) - len(failed)}/{len(results)}")
    if failed:
        print(f"  FAILING PAIRS   : {', '.join(r['concept'] for r in failed)}")
    if drifted:
        detail = ", ".join(
            "{} ({:.3f} vs {})".format(r["concept"], r["normalized"], r["documented"])
            for r in drifted)
        print(f"  DRIFTED FROM DOCS: {detail}")
        print("  -> SUBMISSION_SUMMARY.md and the code disagree. Fix one of them.")
    if not failed and not drifted:
        print("  RESULT: all pairs pass and match the documented figures.")
    print("-" * 78)
    return 0 if not failed and not drifted else 1


if __name__ == "__main__":
    sys.exit(main())
