# Submission summary — Roman Urdu cardiac triage decision support

**Generated 2026-08-16. Paste-ready; every number below is measured, not estimated.**

---

## Paste-ready paragraph

> We present an offline decision-support prototype that triages Roman
> Urdu/English emergency complaints into four acuity levels, restricted in
> scope to **cardiac presentations**. The system contributes (1) an automatic
> stop-word learner for Roman Urdu, which selects tokens by mutual
> information and Cramér's V effect size rather than a chi-square p-value —
> the latter fails to scale, admitting only 10 stop words on a 1.2k corpus
> and vetoing genuinely uninformative filler as the corpus grows — and (2) a
> canonical-form dictionary that maps English and Roman Urdu spellings of the
> same clinical concept onto one token before embedding. On five
> same-meaning complaint pairs written in different languages, mean cosine
> similarity rises from **0.159 on raw text to 0.721 after normalization**
> (+0.563), with all five pairs clearing a 0.5 threshold. The deployed hybrid
> model (attention-weighted bag-of-words + multilingual sentence embeddings +
> structured vitals) reaches **85.15% accuracy with 9.00% under-triage and
> 5.85% over-triage** on a held-out 20% split. **The evaluation dataset is
> synthetic**: complaints were produced by a documented phrase-bank and
> sentence-skeleton generator whose vocabulary derives from an organic corpus
> of 887 distinct cardiac complaints, and triage labels were assigned by
> construction rather than clinician adjudication. No real patient records
> were used. Results characterise the processing pipeline and do not
> constitute clinical validation.

---

## Headline numbers

| metric | value |
|---|---|
| deployed method | D) Hybrid — dictionary BoW + sentence embeddings + vitals |
| accuracy (held-out 20%, n=2000) | **85.15%** |
| under-triage (missed severity) | **9.00%** — safety grade A |
| over-triage | 5.85% |
| classes | 4 (Emergency / Urgent / Standard / Non-Urgent) |
| dataset | `cardiac_multilingual_10000_v3.csv`, 10,000 rows, 10,000 distinct texts |
| operation | fully offline, verified with outbound sockets disabled |

Method comparison on the same split (all four trained identically):

| method | accuracy | under-triage |
|---|---|---|
| A) Dictionary + BoW | 86.35% | 8.45% |
| **D) Hybrid (deployed)** | **85.15%** | **9.00%** |
| C) Embeddings + preprocessing | 82.40% | 11.80% |
| B) Embeddings, raw text | 82.05% | 11.90% |

D was deployed over the marginally more accurate A because it retains the
embedding pathway the study is about while staying inside the <10%
under-triage safety band. C, the previous default, was rejected for
exceeding it.

## What the accuracy number actually measures

The labels are a function of complaint text, ECG status and vitals, sampled
with deliberate overlap so that **no single feature determines the label**.
This matters: an earlier version of this dataset had `ECG_Status` determine
the level almost perfectly (ST elevation → Level 1 in 1,930 of 1,932 rows).
A model on that data scored 98.45%, but structured features alone scored
99.1% — the text pipeline contributed nothing and the headline was an ECG
lookup. The 85.15% reported here is lower and more honest: a bag-of-words
model given only the complaint text scores 69.0% against a 40%
majority-class baseline, confirming the text carries real but partial signal.

## Known limitations (state these before a reviewer finds them)

1. **Synthetic data.** Labels are assigned by construction, not by clinician
   adjudication. Inter-rater agreement is therefore undefined, and no claim
   of clinical validity is made. Full disclosure in `DATASET_PROVENANCE.md`;
   the generator is `generate_cardiac_dataset.py` and the exact file is
   reproducible from its seed.
2. **Synthetic vocabulary ceiling.** The generated corpus has 277 distinct
   words versus 782 in the organic corpus it derives from. A phrase-bank
   generator cannot reproduce the variety of human-written complaints, so
   the text task is easier here than in deployment.
3. **Embedding weakness on some symptom clusters.** On organic reference
   text the multilingual sentence encoder clusters same-meaning complaints
   at 61.9% mean pass rate, but `fracture_sprain` reaches only 35.6% and
   `dizziness_weakness` 44.4%. Dictionary normalization does not move these;
   the limitation is the pretrained model's Roman Urdu coverage. Native-script
   transliteration or a fine-tuned encoder is the plausible remedy and is not
   attempted here.
4. **Cluster metrics are sensitive to corpus repetitiveness.** Measured pass
   rate rises as vocabulary shrinks (95.6% on the most templated corpus,
   79.6% on the organic one), so cluster scores on synthetic text overstate
   real-world separation and should be read alongside the organic-text
   numbers.
5. **Evaluation split is random, not grouped.** Held-out rows are distinct
   texts, but templated generation makes them near-duplicates of training
   rows. A grouped or organic-text evaluation would give a lower and more
   trustworthy estimate.
6. **Cardiac scope only.** Non-cardiac complaints are out of scope; the model
   will still emit a confident level for them. An out-of-scope warning layer
   is designed but not implemented.

## Reproducibility

```bash
python generate_cardiac_dataset.py --out cardiac_multilingual_10000_v3.csv
python train_embedding_pipeline.py --data cardiac_multilingual_10000_v3.csv --deploy D
./run_gui.sh
```

Deployed bundle: `triage_model_embedding/` (snapshot
`triage_model_embedding_v8_v3_deployD/`). The manifest records dataset
filename, row count, sha256, class count, synthetic-data provenance, cardiac
scope and training date; the GUI banner surfaces the synthetic-data warning
on every screen.
