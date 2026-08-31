"""
src/cluster_analyzer.py
=======================================================================
Pairwise similarity over a cluster of complaints.
=======================================================================

Takes a list of complaints, runs each through the offline pipeline, and
reports how tightly the cluster holds together:

    M  =  n x 384 embedding matrix, each row L2-normalised
    S  =  M @ M.T, so S[i][j] is the cosine similarity of i and j

Because every row is L2-normalised, the dot product IS the cosine - no
division needed, and S has an exact 1.0 diagonal that doubles as a
correctness check on the encoder.

WHAT THE NUMBERS ARE FOR
------------------------
Mean intra-cluster similarity says how alike the complaints are. On its
own that is not enough to judge an encoder: a model that maps every
medical sentence to nearly the same vector scores beautifully here while
being useless. Compare against a second cluster - across-cluster
similarity, and the gap between the two - before drawing conclusions.
That caveat is stated in the output rather than left to the reader.

The outlier is the complaint with the lowest mean similarity to the rest.
It is a prompt to look, not a verdict: on a real cluster it is usually
either a genuinely different presentation or a bad translation, and the
two are worth telling apart.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from src.embedding_pipeline import EMBEDDING_DIM, preprocess_and_embed

MATCH_THRESHOLD = 0.85
ALIGNMENT_THRESHOLD = 0.90

#: Ten cardiac/emergency complaints for the GUI preset. Drawn from the
#: phrasing in this project's own datasets so the demo exercises the real
#: vocabulary rather than invented text.
SAMPLE_CLUSTER = [
    "seena mein shadeed dard aur pasina aa raha hai",
    "chest mein pressure aur bhaari pan hai",
    "seena mein jalan aur dard bazu tak ja raha hai",
    "dil ki dhadkan bohat tez ho rahi hai",
    "saans phool rahi hai aur seena tight hai",
    "seedhiyan chadhte waqt seena mein dard hota hai",
    "behoshi jaisa lag raha hai aur thanda pasina",
    "seena mein dabao hai aur jabra mein dard",
    "pait mein dard aur ulti ho rahi hai",
    "sar mein chot lagi hai accident ke baad",
]


def analyze_sentence_cluster(sentences_list: Sequence[str],
                             translate: bool = True,
                             match_threshold: float = MATCH_THRESHOLD,
                             reference: Optional[str] = None,
                             progress=None) -> dict:
    """Embed a cluster and describe how it hangs together.

    `progress(i, n, text)` is called per sentence so a GUI can show
    movement during what is a slow, one-LLM-call-per-sentence job.

    Never raises. A sentence that fails to embed is recorded in `failed`
    and excluded from the matrix, so one bad row cannot destroy the run.
    """
    sentences = [str(s) for s in sentences_list if str(s).strip()]
    result = {
        "n_input": len(sentences_list),
        "sentences": [],
        "matrix": None,
        "mean_similarity": None,
        "top_pairs": [],
        "outlier": None,
        "failed": [],
        "errors": [],
        "encoder": None,
        "threshold": match_threshold,
    }
    if not sentences:
        result["errors"].append("No non-empty sentences given.")
        return result

    rows, kept = [], []
    for i, text in enumerate(sentences):
        if progress:
            try:
                progress(i, len(sentences), text)
            except Exception:
                pass
        try:
            step = preprocess_and_embed(text, translate=translate)
        except Exception as e:
            result["failed"].append({"index": i, "text": text,
                                     "error": f"{type(e).__name__}: {e}"})
            continue
        if step.get("embedding") is None:
            result["failed"].append({"index": i, "text": text,
                                     "error": step.get("error") or "no embedding"})
            continue
        if step.get("error"):
            result["errors"].append(f"[{i}] {step['error']}")
        result["encoder"] = step["encoder"]
        rows.append(step["embedding"])
        kept.append({
            "index": i,
            "raw": step["raw"],
            "translated": step["translated"],
            "normalized": step["normalized"],
            "l2_norm": step["l2_norm"],
            "translated_ok": step["translated_ok"],
            "shape": (EMBEDDING_DIM,),
        })

    if not rows:
        result["errors"].append("Every sentence failed to embed.")
        return result

    M = np.vstack(rows).astype(np.float32)
    # Rows are L2-normalised, so the dot product is already the cosine.
    S = M @ M.T
    np.clip(S, -1.0, 1.0, out=S)
    n = len(M)

    off = ~np.eye(n, dtype=bool)
    result["matrix"] = S
    # The vectors themselves, not just their pairwise scores. A caller that
    # wants to compare a NEW complaint against this cluster would otherwise
    # have to re-translate all ten - one local LLM call each - to rebuild
    # what was already computed here.
    result["vectors"] = M
    result["mean_similarity"] = float(S[off].mean()) if n > 1 else 1.0
    result["min_similarity"] = float(S[off].min()) if n > 1 else 1.0
    result["max_similarity"] = float(S[off].max()) if n > 1 else 1.0
    # Exactly 1.0 down the diagonal, or the vectors are not unit length.
    result["diagonal_ok"] = bool(np.allclose(np.diag(S), 1.0, atol=1e-3))

    pairs = [(float(S[i][j]), i, j)
             for i in range(n) for j in range(i + 1, n)
             if S[i][j] >= match_threshold]
    pairs.sort(reverse=True)
    result["top_pairs"] = [
        {"similarity": sim, "i": i, "j": j,
         "text_i": kept[i]["raw"], "text_j": kept[j]["raw"]}
        for sim, i, j in pairs
    ]

    if n > 1:
        means = (S.sum(axis=1) - 1.0) / (n - 1)
        worst = int(np.argmin(means))
        result["per_sentence_mean"] = [float(x) for x in means]
        result["outlier"] = {
            "index": kept[worst]["index"],
            "text": kept[worst]["raw"],
            "translated": kept[worst]["translated"],
            "mean_similarity": float(means[worst]),
            "note": ("Lowest mean similarity to the rest. Usually either a "
                     "genuinely different presentation or a bad translation - "
                     "worth reading before treating it as either."),
        }

    if reference:
        try:
            ref = preprocess_and_embed(reference, translate=False)
            if ref.get("embedding") is not None:
                sims = M @ ref["embedding"]
                result["reference"] = {
                    "text": reference,
                    "similarities": [float(x) for x in sims],
                    "mean": float(np.mean(sims)),
                    "aligned": [bool(x >= ALIGNMENT_THRESHOLD) for x in sims],
                    "threshold": ALIGNMENT_THRESHOLD,
                }
        except Exception as e:
            result["errors"].append(f"reference embedding failed: {e}")

    result["sentences"] = kept
    return result


def format_matrix(S, labels=None, width: int = 6) -> str:
    """Plain-text square matrix, for console use."""
    if S is None:
        return "(no matrix)"
    n = len(S)
    labels = labels or [f"S{i+1}" for i in range(n)]
    head = " " * 6 + "".join(f"{l:>{width}}" for l in labels)
    lines = [head, " " * 6 + "-" * (width * n)]
    for i in range(n):
        lines.append(f"{labels[i]:>5} " +
                     "".join(f"{S[i][j]:>{width}.2f}" for j in range(n)))
    return "\n".join(lines)
