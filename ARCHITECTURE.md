# ARCHITECTURE.md — Embedding-Based Triage Pipeline (spec for implementation)

This document describes the target architecture and the tasks required to get
there. It is written so an engineer (or Claude Code) can implement it against
the existing codebase. It is based on the supervisor's ("Sir") whiteboard
design dated 3.8.26 and the team discussion.

Context: a paper reviewer flagged the project's **contribution as insufficient**.
Two new contributions are being added to strengthen it:
- **Contribution 1:** an *automatic* stop-word learner for Roman Urdu.
- **Contribution 2:** a quantitative *evaluation of the embedding generator's
  effectiveness* (how faithfully it represents complaints).

---

## 1. Target architecture

```
   Structured input                 Unstructured input
   (age, vitals, ECG, AVPU)         (complaint text, Roman Urdu)
          |                                   |
          |                                   v
          |                          [ Preprocessing ]
          |                          - auto stop-word removal (Contribution 1)
          |                          - fuzzy spelling normalization (bukhar/bukhaar)
          |                                   |
          |                                   v
          |                          [ Sentence Transformer ]
          |                          - offline embedding model (CPU)
          |                                   |
          |                                   v
          |                          [ Embedding vector ]  (array of N numbers)
          |                                   |
          +----------------> [ Fuse ] <-------+
                             (concatenate text embedding + structured features)
                                     |
                                     v
                             [ Classifier ]
                             (Logistic Regression)
                                     |
                                     v
                             Triage level (1 = emergency ... 4 = non-urgent)
```

Key change from the current system: the complaint text is turned into numbers by
an **offline sentence-embedding model** instead of the hand-built dictionary +
Bag-of-Words. The dictionary's fuzzy step is *kept* inside preprocessing.

> Embedding dimension is decided by the model chosen (e.g. ~384 for
> MiniLM-class models). The whiteboard's "364" is a placeholder; use whatever
> the selected model outputs.

---

## 2. Current state (what already exists)

Existing files this builds on:

- `triage_pipeline.py` — shared module: text normalization, fuzzy matching,
  diacritization, Bag-of-Words + attention, structured-feature helpers,
  `predict_*` functions, artifact loading.
- `train_embedding_pipeline.py` — training script for the deployed bundle.
- `predict_batch.py` — file-based inference.
- `run_inference.py` — CLI over the offline Ollama pipeline.
- `src/offline_pipeline.py` — prompt, translation, refusal guardrail, fuzzy
  dictionary, anatomical assertion gate. This is where the serving path lives.
- `embedding_evaluation.py` — compares dictionary vs embeddings vs both,
  using `sentence-transformers`. **This is the starting point for the embedding
  work** — it already loads an offline model, encodes complaints, fuses with the
  structured features, and trains the same classifier.
- `triage_mixed_language_dataset.csv` — 1,204 labelled patients.

Reuse `train_embedding_pipeline.py`'s encoding + fuse + train logic rather than
writing it again.

---


### The serving path today

The document below describes the embedding pipeline as designed. What actually
serves a patient now has two stages in front of it and one after:

```
raw Roman Urdu
  1. fuzzy_normalize_roman_urdu()   src/offline_pipeline.py  - local, deterministic
  2. translate_roman_urdu()         Ollama on localhost, llama3.2, temperature 0
  3. verify_anatomical_integrity()  deterministic gate - THIS is the safety check
  4. build_text_features()          triage_pipeline.py - stop words + encoder
  5. predict_proba -> argmax        confidence is the winning class probability
```

Three things this changed, all of them load-bearing:

- **The gate replaced cosine similarity as the decision.** Cosine is still
  measured and reported and decides nothing: a correct translation scores
  0.8054 and *"My leg is broken after a fall"* scores 0.7922, so no threshold
  separates them.
- **The GUI has one pipeline.** The mode toggle is gone; it serves
  `triage_model_embedding_english/` (2,252 rows), not the 10,000-row bundle.
- **The dictionary runs before the LLM, not instead of it.** Variants collapse
  onto a canonical Roman Urdu token so the model sees one spelling per word.

## 3. Task list

### Task 1 — Automatic stop-word removal  (Contribution 1)

Goal: a preprocessing step that **learns** which words are stop words from the
training data, instead of using a fixed hand-written list.

Definition of a stop word here: a token that appears very frequently across
complaints of *all* triage levels and therefore carries little signal about
urgency (e.g. `hai`, `hain`, `ho`, `ga`, `ke`, `aur`).

Suggested method (data-driven, explainable — good for a paper):
1. Tokenize all complaints in the training set.
2. For each token compute:
   - **document frequency** (fraction of complaints it appears in), and
   - a **class-discrimination score** — e.g. mutual information or chi-square
     between "token present" and the triage label.
3. Mark a token as a stop word if its document frequency is high (say top X%)
   **and** its discrimination score is near zero (it does not help predict the
   triage level).
4. Save the learned stop-word list to a file (e.g. `learned_stopwords.json`) so
   it is reproducible and can be shown in the paper.
5. Remove those tokens during preprocessing, before embedding.

Deliverables:
- A function `learn_stopwords(corpus, labels) -> set[str]` (new module, e.g.
  `stopwords.py`, or added to `triage_pipeline.py`).
- A `remove_stopwords(text, stopword_set) -> str` used in preprocessing.
- The saved `learned_stopwords.json` for the paper's methods section.

> Keep it a *threshold-based, transparent* method. Reviewers like being able to
> see exactly why each word was dropped. Avoid a black-box approach here.

### Task 2 — Keep fuzzy spelling cleaning

The fuzzy matching that collapses `bukhar / bukhaar / bukharr` into one form
already exists in `triage_pipeline.py`. Keep it as part of preprocessing (it
handles "more than one spelling of the same word"). Preprocessing order:

```
raw text -> lowercase/clean -> fuzzy normalize -> remove learned stop words -> embed
```

### Task 3 — Embedding text representation + fuse + classify

Wire the pipeline so preprocessed complaint text goes through the sentence
transformer, and the resulting embedding is concatenated with the structured
features and fed to the classifier. `embedding_evaluation.py`
already does the encode + fuse + train; promote that path into the main training
flow once Task 1's preprocessing is in front of it.

Run the three-way comparison (dictionary / embeddings / both) again *after*
adding automatic stop-word removal, to see if preprocessing improved the
embedding result. Keep whichever scores best on accuracy and under-triage.

### Task 4 — Embedding-effectiveness study  (Contribution 2)

This is a separate analysis script (suggested name: `embedding_evaluation.py`).
It does NOT change the classifier — it measures how good the embedding generator
is at representing complaints.

**4a. Manual clusters.** Create ~10 clusters of ~10 complaints each, grouped by
meaning: chest pain, unconsciousness, hand pain, abdominal pain, breathing
difficulty, fever, bleeding, seizure, burn, etc. Store them in an editable file
(e.g. `evaluation_clusters.json`) so the team can adjust membership.

**4b. Within-cluster correlation.** For each cluster, embed all its complaints
and compute the **pairwise similarity between every pair**. For a cluster of 10
that is exactly:

```
C(10,2) = 10! / (2! * 8!) = 45 pairs
```

Report, per cluster:
- all 45 pairwise **correlation / cosine similarity** values (range 0–1),
- the pairwise **Euclidean distance** values,
- the fraction of pairs above the **0.5** threshold (Sir's cut-off for "these
  two complaints are correctly seen as similar").

A good embedding generator gives high within-cluster similarity (most pairs
> 0.5) and lower across-cluster similarity.

**4c. Round-trip fidelity test ("does it regenerate the same text?").**

> IMPORTANT — read before implementing. Sir's description is "convert text to
> embedding, then convert the embedding back to text and check it is the same."
> Standard sentence embeddings are **one-way / lossy**: you cannot feed a vector
> back into the model and recover the exact original sentence, because that
> information is not stored. Do NOT attempt literal embedding inversion (it needs
> special models, is English-only, and will stall the project).
>
> The correct, implementable version that measures the same thing is a
> **nearest-neighbour round-trip**:
> 1. Embed every complaint in the dataset.
> 2. For each complaint's embedding, search all embeddings for the **closest
>    one that is not itself**.
> 3. Check whether that nearest neighbour is a complaint from the **same manual
>    cluster** (i.e. the same meaning).
> 4. **Efficiency score = fraction of complaints whose nearest neighbour is a
>    correct same-meaning complaint.** Higher = the embedding generator
>    faithfully represents the text. This is the number to report in the paper
>    as "embedding generator efficiency."

Deliverables:
- `embedding_evaluation.py` producing: per-cluster similarity tables, the 45
  pairwise values per cluster, the > 0.5 pass rate, and the overall
  nearest-neighbour efficiency score.
- A results CSV (e.g. `embedding_evaluation_results.csv`) for the article.

### Task 5 — Research (non-code)

Study the "Results" sections of 5–10 related research papers to match how the
embedding-effectiveness numbers should be presented in the article.

---

## 4. Honest engineering notes (raise these with Sir)

1. **Roman Urdu is the main risk.** Sentence-embedding models are trained mostly
   on English and native-script Urdu (اردو). Romanized Urdu ("dard", "drd") is
   under-represented, so within-cluster similarity may be lower than hoped. The
   evaluation in Task 4 will reveal this directly. If similarity is weak, options
   are: (a) transliterate Roman Urdu to native script before embedding, or
   (b) fine-tune a small model on your own data (only if Sir approves — larger
   task).

2. **Lossy embeddings.** See Task 4c — the literal "regenerate the text" test is
   not possible with normal embeddings; use nearest-neighbour fidelity.

3. **Lightweight tension.** The project's selling point is small + offline +
   CPU-only. An embedding model adds a few hundred MB. Worth confirming with Sir
   that this trade-off is acceptable, or keeping a hybrid (dictionary + embedding)
   so the dictionary still anchors the Roman Urdu the model misses.

4. **Keep everything offline.** All of this must run without internet after a
   one-time model download. `sentence-transformers` caches the model locally and
   then runs offline, which satisfies the project's constraint.

---

## 5. Suggested new / changed files

| File | Change | Task |
|---|---|---|
| `stopwords.py` (or added to `triage_pipeline.py`) | NEW: learn + remove stop words | 1 |
| `learned_stopwords.json` | NEW: saved output for the paper | 1 |
| `triage_pipeline.py` | preprocessing = fuzzy + stop-word removal | 1, 2 |
| main training flow | use embedding path in front of classifier | 3 |
| `evaluation_clusters.json` | NEW: 10 manual clusters, editable | 4 |
| `embedding_evaluation.py` | NEW: correlations (45/cluster), Euclidean, NN fidelity | 4 |
| `embedding_evaluation_results.csv` | NEW: results for the article | 4 |

---

## 6. Open questions to confirm with Sir

- Which embedding model / dimension does he want standardized on?
- Is the round-trip test acceptable as nearest-neighbour fidelity (given true
  inversion isn't possible)?
- Should the final system be embeddings-only, or hybrid (dictionary + embeddings)?
- Does the size increase from the embedding model conflict with the "lightweight"
  claim in the paper, and how should that be framed?
