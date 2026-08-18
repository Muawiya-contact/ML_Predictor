# `src/` — dual-pipeline classification for Roman Urdu complaints

Two targets (`Triage_Level`, `Category`/department) over one embedding space,
with an offline evaluation path and a live inference path that are held to the
same representation.

```
src/encoders.py   StaticEncoder (.npy)  |  DynamicEncoder (live text)
src/models.py     LogReg / RandomForest / HistGradientBoosting + metrics
src/baseline.py   stratified 5-fold CV over both targets
src/train.py      refit on all rows, persist to models_src/
src/predict.py    raw complaint -> triage level + department
```

```bash
.venv/bin/python -m src.baseline                 # cross-validated benchmark
.venv/bin/python -m src.train                    # persist production models
.venv/bin/python -m src.predict "Mera sar dard ho raha hai"
.venv/bin/python -m src.predict --interactive
```

## Why these three classifiers

XGBoost and LightGBM are deliberately **not** dependencies.
`HistGradientBoostingClassifier` is scikit-learn's gradient booster and gives
the benchmark a boosted-tree entry with nothing to install on a machine that
has to run offline. On 185 rows the gap between it and XGBoost is far smaller
than the fold-to-fold variance, which the reported standard deviations make
visible.

`early_stopping` is off on the booster. With it on, the estimator carves its
own stratified validation slice out of each training fold, and
`Metabolic_Cardiovascular` has **2 rows in the entire dataset** — once
cross-validation puts one in test, the inner split sees a single member and
sklearn raises *"The least populated classes in y have only 1 member"*.

## Cross-validated results (5-fold, seed 42, 185 rows)

**Triage_Level** — 4 classes, support `{1: 58, 2: 48, 3: 57, 4: 22}`

| model | accuracy | macro-F1 | under-triage | grade |
|---|---|---|---|---|
| LogisticRegression | 73.5% ± 8.6 | 0.695 ± 0.104 | 11.9% ± 3.7 | B |
| **RandomForest** | **80.5% ± 7.1** | **0.783 ± 0.093** | **6.5% ± 2.8** | **A** |
| HistGradientBoosting | 72.4% ± 11.4 | 0.693 ± 0.136 | 10.3% ± 4.3 | B |

**Category (department)** — 11 classes

| model | accuracy | macro-F1 |
|---|---|---|
| LogisticRegression | 91.9% ± 1.7 | 0.873 ± 0.016 |
| **RandomForest** | **94.6% ± 1.7** | **0.909 ± 0.082** |
| HistGradientBoosting | 88.1% ± 4.4 | 0.841 ± 0.092 |

Department routing scores far higher than triage, which is expected: the
complaint text names the body system directly, while acuity depends on
severity and vitals the text only partly carries.

## Findings on the supplied baseline

Recorded as observations, not debugged — the baseline scripts were left
untouched.

**1. Triage and department results are byte-identical between the translated
and direct-Roman-Urdu runs.** `triage_classifier_results.xlsx` and
`triage_classifier_results_direct_roman_urdu.xlsx` agree to every decimal on
precision, recall, F1 and support, and so do the two department workbooks.
Either translation changed nothing at all, or the same predictions were
written to both files. As it stands the headline "translation vs no
translation" comparison shows no difference, so neither file should be cited
as evidence for that comparison until it is resolved.

**2. The reported 97.3% triage accuracy rests on a 37-row test set.** Four of
those rows are Level 4. One flipped prediction moves accuracy ~2.7 points and
that class's F1 by far more. The 5-fold numbers above are lower and carry
standard deviations for exactly this reason.

**3. Two shipped `.npy` files are empty.** `roman_urdu_embeddings.npy` and
`translated_embeddings.npy` are 128-byte headers with shape `(0,)` — whatever
wrote them failed silently. `StaticEncoder` raises on an empty matrix rather
than letting a zero-row array reach a classifier.

**4. `ground_truth_embeddings.npy` is 768-dimensional**, not 384, so it comes
from a different encoder than `complaint_embeddings.npy` and the two are not
comparable without re-encoding.

## Encoder contract

Both encoders commit to the settings the stored vectors were built with, read
out of the baseline's own scripts rather than guessed:

| | value |
|---|---|
| model | `intfloat/multilingual-e5-small` |
| dims | 384 |
| prefix | `"passage: "` on every complaint |
| normalization | L2 |

The prefix is not cosmetic — e5 is trained with `query:`/`passage:` markers,
and dropping it at inference while the stored vectors carry it shifts the
whole space. `predict.py` checks the manifest against `src/encoders.py` and
refuses to run on a mismatch, because a classifier fed the wrong embedding
space does not crash, it just answers worse.

## Limitations

- **185 rows.** Every number here has wide error bars; the standard deviations
  are the honest part of the table.
- **`Metabolic_Cardiovascular` has 2 rows**, `Non_Specific` has 5. Their
  per-class scores are decided by one or two test rows and are not
  performance measurements.
- Train accuracy printed by `src/train.py` is fitted-on-itself and will read
  near 100%. The cross-validated numbers are the ones to quote.
- Research prototype. Not a medical device.
