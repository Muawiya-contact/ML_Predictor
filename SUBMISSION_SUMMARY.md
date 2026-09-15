# Submission summary: Roman Urdu medical triage in Pakistan

The article configuration is C: preprocessed English sentence embeddings fused
with structured features, classified by Logistic Regression. The active bundle
is `triage_model_embedding_english/`, trained on 2026-09-09. This cleanup retains
its existing weights and recorded results; it does not retrain the classifier.

## Methodology points

1. Frame the study as an offline research prototype for Roman Urdu cardiac triage.
2. Disclose the synthetic dataset: 2,252 translated cardiac records, not patient data.
3. Describe the fuzzy clinical dictionary and local Ollama translation at temperature 0.0.
4. Describe refusal handling and the deterministic body-part consistency check.
5. Explain statistical stop-word selection, clinical-term protection and saved reports.
6. Name the multilingual MiniLM checkpoint and its 384-dimensional normalized vectors.
7. Describe the six numeric inputs and four categorical fields, producing 26 structured columns.
8. Concatenate these with embeddings to form 410 classifier inputs; use balanced Logistic Regression.
9. Report the stratified 80/20 split (1,801/451), seed 42, accuracy, under/over-triage and confusion matrix.
10. Present raw versus preprocessed embedding ablation and separate semantic-cluster diagnostics.

| Recorded configuration | Accuracy | Under-triage | Over-triage |
|---|---:|---:|---:|
| B: raw English embeddings | 80.71% | 13.30% | 5.99% |
| C: preprocessed English embeddings | 80.49% | 12.64% | 6.87% |

Preprocessing is associated with 0.66 percentage points less under-triage and
0.22 points less accuracy in this split. This is not evidence of a general
accuracy improvement or clinical safety.

## Limits that belong in the article

Training translations came from gpt-4o-mini, while serving uses local Ollama.
The classifier scores therefore do not establish live translation performance.
The dataset is synthetic and cardiac-only. The historical trainer fits structured
preprocessing before splitting and exports full-data stop words alongside the
split-trained classifier. The saved classifier and those preprocessing choices
must be described accurately, with further evaluation identified as future work.

The professor baseline is a separate 185-row study with e5-small embeddings,
two targets and stratified five-fold cross-validation. Do not attribute its
protocol, classifiers or results to the GUI experiment. Nothing in this project
constitutes clinical validation of a medical device.

Dataset identity: `cardiac_english_2252.csv`, SHA-256
`42ab09e252db94b8f7b0dea38fae47556f8d38c380fdc43fdbb37f2ea728cb1c`.
The authoritative recorded results and configuration are in the active bundle's
`triage_metrics.json` and `model_manifest.json`.
