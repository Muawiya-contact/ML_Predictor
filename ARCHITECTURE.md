# Embedding-based triage architecture

## Serving path

Roman Urdu complaint -> fuzzy spelling normalization -> local Ollama English
translation -> refusal filtering -> deterministic anatomical gate -> learned
English stop-word removal -> MiniLM sentence embedding -> concatenation with
structured features -> Logistic Regression -> triage level 1-4.

`triage_model_embedding_english/` is the active 410-feature bundle:
26 structured features followed by 384 L2-normalized embedding dimensions.
Its manifest records configuration C, English_Translation as the text column,
and skip_normalization=true. English is not passed back through the Roman Urdu
dictionary before embedding. The existing model weights are retained.

## Module responsibilities

- `triage_pipeline.py`: normalization, protected clinical vocabulary, embedding
  bundle validation/loading, feature assembly, missing-input handling and scoring.
- `src/offline_pipeline.py`: Ollama selection and translation, refusal handling,
  pre-translation fuzzy matching and deterministic anatomical verification.
- `triage_gui.py`: six-tab desktop workflow, translation/gate feedback, single
  and batch triage, results and diagnostic views.
- `predict_batch.py`: file-based CLI, translating and gating rows before scoring
  with the English model; rejected rows have no prediction.
- `train_embedding_pipeline.py`: raw (B) and preprocessed (C) embedding comparison,
  concatenation with structured features, classifier fitting and bundle export.
- `stopwords.py`: document frequency, normalized mutual information and Cramer's V
  selection, clinical-term protection and inspectable per-token reports.
- `embedding_evaluation.py`: cluster geometry and nearest-other-complaint agreement.
- `src/embedding_pipeline.py` and `src/cluster_analyzer.py`: diagnostic embedding
  and similarity functions, with their own documented preprocessing.

## Model contract

Every bundle requires a manifest, classifier, numeric scaler, four categorical
encoders and a bundle-local stop-word list. Only embeddings_raw and
embeddings_preprocessed representations are accepted. Feature order and classifier
input width are checked when loading. Missing dependencies or artifacts must
produce a clear failure, not an unannounced substitution.

## Evaluation and research limits

The active classifier's recorded experiment uses 2,252 synthetic cardiac rows,
an 80/20 stratified split and seed 42. Recorded C results are 80.49% accuracy,
12.64% under-triage and 6.87% over-triage. The English training translations
were prepared with gpt-4o-mini; serving uses Ollama. These classifier results
are not an evaluation of the complete live translation pipeline.

Evaluation stop words are learned on training rows. The historical trainer fits
structured preprocessing before splitting, then exports a full-data stop-word
list with the split-trained classifier. These limitations need to be addressed
in a separately evaluated experiment before claiming leakage-free end-to-end
validation. Do not retrain the shipped bundle merely to fix an installation.

## Separate baseline and embedding research

The 185-row professor baseline uses e5-small embeddings and stratified five-fold
evaluation of Triage_Level and Category. `models_src/` supplies two RandomForest
classifiers. `src/models.py` also benchmarks Logistic Regression and
HistGradientBoosting. `run_inference.py` runs this text-only baseline.
Its 384-dimensional vectors are not interchangeable with the GUI's MiniLM space.

The fuzzy dictionary and automatic stop-word learner remain research components.
Nearest-neighbour agreement is a proxy for semantic preservation, not a decoder
that regenerates the original complaint. Clinical evaluation on independently
collected, clinician-labelled data remains future work.
