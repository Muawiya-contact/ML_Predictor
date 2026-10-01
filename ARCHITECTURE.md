# Four-level SapBERT triage architecture

## Serving path

Roman Urdu complaint → fuzzy normalization → local Ollama translation → refusal
filter and deterministic anatomical gate → SapBERT CLS (768-D, L2-normalized)
→ fitted PCA (64-D) → concatenate 22 patient features → CV-selected classifier
→ level 0 Emergency, 1 Urgent, 2 Standard or 3 Non-urgent.

The current bundle is `triage_model_sapbert/`. `model_manifest.json` identifies
its exact checkpoint, classifier configuration, four-class mapping, dataset,
selection rule and artifact checksums. `triage_metrics.json` describes that same
fitted model. No model is refitted or substituted during inference.

## Modules

- `triage_pipeline.py`: common manifest handling and prediction entry points.
- `src/sapbert_serving.py`: offline CLS encoder, numeric/AVPU conversion,
  fitted preprocessing and four-level probabilities; validates saved artifacts.
- `src/offline_pipeline.py`: local translator selection, fuzzy repair, refusal
  filtering and deterministic anatomical consistency checks. Its `run` function
  remains a separate historical professor-baseline interface.
- `triage_gui.py`: asynchronous desktop prediction, six tabs, batch gate status,
  input-change invalidation and active-model results.
- `predict_batch.py`: translation-aware file prediction with blank rejected rows.
- `run_inference.py`: single/interactive CLI using the current fused bundle.
- `src/cluster_analyzer.py`: pairwise diagnostics; the GUI supplies its active
  SapBERT encoder and uses vectors before PCA.
- `experiments/triage_study/four_level_study.py`: grouped CV, fixed 12-condition
  comparison, frozen selection, holdout evaluation and fitted artifacts.
- `prepare_four_level_source.py`, `verify_four_level_study.py` and
  `export_four_level_bundle.py`: source recovery, independent result verification
  and export with full holdout-prediction parity.

## Data and inference contracts

Training consumes Clinical_Concept plus six numeric patient measurements,
ordinal AVPU and three categorical fields. Target names and processing metadata
are excluded. Imputation, scaling, one-hot encoding and PCA are fitted only on
training rows within each fold. Training text is supplied/recovered concepts;
live text is Ollama English. Their evaluation scopes remain distinct.

The GUI clears stale predictions when inputs change. Blank/placeholder text
shows 50% with no level and an explicit placeholder explanation. Translation or
anatomical failures withhold a score. Classifier probabilities are not clinical
certainty. Raw source records are local; published results contain aggregates.

See [the current protocol](experiments/triage_study/FOUR_LEVEL.md),
[GUI guide](docs/SapBERT_GUI.md) and [report directory](reports/triage_four_level/).
Historical MiniLM/three-level/professor experiments use different feature spaces
and labels; their scores are not the current application's scores.
