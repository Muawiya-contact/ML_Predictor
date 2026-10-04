# Current four-level SapBERT application

Run `./run_gui.sh` (or `./run_local.sh` to check/start Ollama). GUI, batch CLI and
`run_inference.py` default to `triage_model_sapbert/`. Its manifest identifies the
classifier selected by five-fold grouped development cross-validation: balanced
Logistic Regression C=10. The four fitted preprocessing/model artifacts remain
synchronized through manifest checksums. All live
levels use the workbook mapping: **0 Emergency, 1 Urgent, 2 Standard, 3 Non-urgent**.

SapBERT uses CLS pooling, L2-normalized 768-dimensional vectors and a 64-token
limit. The fitted PCA projects these to 128 dimensions; 50 structured features
are concatenated (35 numeric linear/quadratic terms and 15 categorical indicators),
plus 19 explicit details from the original complaint, for 197 total inputs. No stop words are removed for this bundle. Similarity and
cluster panels use the full 768-dimensional vectors before PCA.

The six tabs share the active encoder and manifest. Results reads the selected
bundle's saved four-class confusion matrix. Single-patient translation runs in
a background worker; editing patient inputs clears the old result and discards
an outstanding result for outdated inputs. Empty or placeholder complaints show
a yellow **Confidence: 50%** banner with no level or probability bars. The text
panel explains that this is a display placeholder, not a model prediction.

Local Ollama translation, refusal checks and the deterministic anatomical gate
precede classification. A cold Qwen prompt can take several minutes on a busy
CPU; the GUI allows up to 15 minutes for the response. Translation failures
and anatomical mismatches are not assigned a level. Missing structured values
are imputed using the fitted training statistics with a visible note.

## Model and report identity

The current workbook is `cardiac_multilingual_10000_4level_triage.xlsx`.
The evaluation uses its supplied/recovered clinical concepts; the live app embeds
Ollama translations. The reported classifier scores are therefore not measured
end-to-end translation scores or clinical validation. The provider described
label assignment as “by using all”; the exact process and independent per-record
review are not documented. No clinician-review claim is inferred from that reply.

The model is fitted on development rows only. The held-out rows are not used to
choose settings or to refit the deployed classifier. The PDF, exported metrics
and application all describe that same selected model. See
[the four-level protocol](../experiments/triage_study/FOUR_LEVEL.md).

## Local encoder setup

The manifest records the pinned SapBERT revision. Copy that snapshot into the
local cache, or set `SAPBERT_MODEL_PATH` to its directory. The application can
also read the pinned revision from the standard Hugging Face cache. It never
downloads a model during SapBERT prediction and does not substitute MiniLM.
For a one-time download before offline use:

```bash
hf download cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
  --revision 090663c3ae57bf35ffe4d0d468a2a88d03051a4d
python run_inference.py --check
```

`TRIAGE_MODEL_DIR` explicitly selects a compatible bundle. The current SapBERT
loader verifies all four fitted artifacts against manifest checksums and
rejects the previous three-class bundle. Use only trusted pickle artifacts.

To export a completed study to a new empty directory without retraining:

```bash
python experiments/triage_study/export_improved_bundle.py \
  --source output/results/triage_four_level_round5 \
  --original output/results/triage_four_level \
  --incumbent output/results/four_level_round4_bundle \
  --output /tmp/four_level_bundle
```

The exporter reproduces every held-out prediction before writing the bundle.

Direct `predict_one` callers must pass `raw_complaint`; direct dataframe callers
must preserve `Raw_Complaint`. GUI and translated batch/CLI wrappers supply it
automatically. Translated English is not substituted for original detail inputs.

The [expanded classifier comparison](../reports/triage_four_level_round6/)
retains this same verified model. CatBoost, XGBoost and SVM were tested as
challengers; the GUI does not silently switch to a lower-scoring candidate.
Their optional experiment packages are unnecessary for the retained LR bundle.
