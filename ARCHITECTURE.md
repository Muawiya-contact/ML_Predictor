# Four-level SapBERT triage architecture

## Serving path

Roman Urdu complaint → fuzzy normalization → local Ollama translation → refusal
filter and deterministic anatomical gate → English + [SEP] + original complaint → SapBERT CLS (768-D, 128-token limit, L2-normalized)
→ fitted PCA (64-D) → concatenate 50 patient features (including quadratic numeric terms) → append 19 original-complaint details → CV-selected classifier
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

- `improve_four_level.py`, `improvement_interactions.py` and `finalize_improvement.py`:
  the original development search and joint selection.
- `refine_complaint_details.py`, `verify_detail_refinement.py`, `investigate_four_level_errors.py`:
  current-data error diagnostics and 13 further settings, bringing the comparison
  to 68 settings and 340 full-size grouped-fold fits.
- `expanded_classifiers.py`: eleven additional CatBoost, CPU XGBoost and RBF SVM
  settings; the combined comparison has 79 settings and 395 grouped-fold fits.
  That historical expansion retained the prior Logistic Regression.
- `embedding_diagnostics.py`: development-only descriptive geometry by triage label.
- `verify_improvement.py`, `export_improved_bundle.py`, `verify_improved_serving.py`:
  independent metric verification and full live-adapter prediction parity.

- `advanced_classifiers.py`, `paired_classifiers.py`, `probability_blends.py`: ordered/neural/nonlinear models, paired text and fixed-weight comparisons.
- `finalize_advanced.py`: verifies all follow-up OOF probabilities and selects before retrospective evaluation.
- `src/ordinal_classifier.py`, `src/soft_voting.py`: ordered and ensemble research candidates. The selected deployment remains paired-text Logistic Regression C=100.

## Data and inference contracts

Training consumes Clinical_Concept + [SEP] + the original complaint, plus six numeric patient measurements,
ordinal AVPU, three categorical fields and 19 original-complaint details. Target names and processing metadata
are excluded. Imputation, scaling, one-hot encoding and PCA are fitted only on
training rows within each fold. Training text is supplied/recovered concepts;
live text is Ollama English. Their evaluation scopes remain distinct.

The GUI clears stale predictions when inputs change. Blank/placeholder text
shows 50% with no level and an explicit placeholder explanation. Translation or
anatomical failures withhold a score. Classifier probabilities are not clinical
certainty. Raw source records are local; published results contain aggregates.

See [the current protocol](experiments/triage_study/IMPROVEMENT.md),
[GUI guide](docs/SapBERT_GUI.md) and [report directory](reports/triage_four_level_round7/).
Historical MiniLM/three-level/professor experiments use different feature spaces
and labels; their scores are not the current application's scores.
