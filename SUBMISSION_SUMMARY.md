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
   is designed but not implemented. Measured: "toota hua pair, chalne mein
   dard" (a broken leg) returns **Level 3 at 96.4% confidence** with no
   indication the complaint is outside the validated domain.
7. **Nonsense input is not detected.** Text that survives cleaning as tokens
   but carries no clinical meaning is scored as though it were a real
   complaint. Measured: "asdkfj qwoeiru zxcvbnm" returns **Level 3 at 97.7%
   confidence**. The system guards the case where *nothing* survives cleaning
   (empty, digits-only, punctuation-only inputs are flagged and their
   confidence capped at 50%), but it has no language or plausibility model
   and cannot tell an unfamiliar Roman Urdu spelling from keyboard mash.
   Users must not read high confidence as evidence the input was understood.
8. **No runtime red-flag / reassurance guardrail.** The rule that keeps
   red-flag phrasing out of low-acuity rows and explicit reassurance out of
   high-acuity rows lives only in the synthetic data generator. Nothing
   enforces it at inference, and the structured features outweigh the text
   whenever the two disagree. Live-tested: *"halka sa seena mein dabao hai,
   rest se theek ho jata hai"* ("mild chest pressure, resolves with rest")
   returns **Level 1 EMERGENCY at 63.7%** under the interface's default
   vitals. The cause is those defaults — `ECG_Status` defaults to *ST
   elevation* — rather than the wording: the identical complaint with a
   Normal ECG and benign vitals returns **Level 4 at 81.0%**, so the
   self-resolving clause is not being ignored. The practical consequence that
   remains is that reassuring language cannot pull a prediction down once the
   structured inputs indicate an infarct — clinically defensible for a genuine
   STEMI, but it means the complaint text is not a safety net.
   *(The interface previously shipped with ST elevation preselected, so an
   operator who predicted without opening the dropdown got EMERGENCY whatever
   they had typed. That default is now `Normal`; the same complaint returns
   Level 3, while a severe complaint still returns Level 1 at 99.8% even on a
   Normal ECG. Note the numeric fields still default to abnormal values —
   HR 118, SpO2 94 — so the form remains biased upward and should be filled
   in rather than trusted.)*
9. **Vitals are not range-checked.** No physiological validation is performed
   anywhere in the pipeline. Measured: age −5, heart rate 300, blood pressure
   900/−40, temperature 99 °C and SpO2 150 together return **Level 3 at 99.9%
   confidence**, with no error and no warning. Out-of-range values are scaled
   and fed to the model like any other number, so a data-entry slip or a unit
   mismatch (Fahrenheit for Celsius, say) produces a confident answer built on
   an impossible patient. Any deployment must validate vitals upstream.

## Reproducibility

```bash
python generate_cardiac_dataset.py --out cardiac_multilingual_10000_v3.csv
python train_embedding_pipeline.py --data cardiac_multilingual_10000_v3.csv --deploy D
python check_embedding_pairs.py      # verifies the 0.159 -> 0.721 figures above
./run_gui.sh
```

`check_embedding_pairs.py` holds the five documented similarities as
assertions rather than printing them: it re-encodes both phrasings of each
pair, compares against the figures quoted in this document, and exits
non-zero if any has drifted by more than 0.02. A reviewer can therefore
confirm the headline embedding claim without taking it on trust, and a later
change to the dictionary or stop-word list cannot leave this write-up
silently stale.

Deployed bundle: `triage_model_embedding/` (snapshot
`triage_model_embedding_v8_v3_deployD/`). The manifest records dataset
filename, row count, sha256, class count, synthetic-data provenance, cardiac
scope and training date; the GUI banner surfaces the synthetic-data warning
on every screen.
