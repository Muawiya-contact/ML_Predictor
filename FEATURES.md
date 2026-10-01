# Current application features

The desktop application contains six tabs and reads the current four-level
SapBERT bundle. Its title/provenance panel identifies the model and evaluation
scope. The class mapping is 0 Emergency, 1 Urgent, 2 Standard, 3 Non-urgent.

## Triage a Patient

Enter a Roman Urdu or English complaint and patient measurements: age, heart
rate, systolic/diastolic blood pressure, temperature, oxygen saturation, gender,
arrival mode, AVPU and ECG status. Local Ollama translates the complaint and a
deterministic anatomical gate checks it. The selected fused PCA-128 classifier
returns a level and four probability bars. The text panel explains each stage.

Prediction runs asynchronously. Changes to any patient input clear a previous
result and invalidate outstanding work for old inputs. Missing/placeholder text
shows 50% without a level; the explanation identifies this as a display
placeholder rather than a prediction. Translation/gate failures are not scored.
Optional local speech reads the translated complaint.

## Pipeline Explorer

Inspect normalized input, local English translation, gate outcome and the
active encoder. SapBERT creates 768-D normalized CLS vectors; the classifier
uses fitted PCA-128 plus 50 patient features (including quadratic numeric terms)
and 19 explicit original-complaint details, for 197 classifier inputs. Similarity views use full vectors.

## Stop Words

Displays the active bundle's filtering information. The current SapBERT model
uses unfiltered English, so its learned stop-word list is empty. Historical
stop-word research does not alter the current classifier's inputs.

## Batch File

CSV/XLSX inputs are translated and gated per row. Exports include original text,
translation, stage details, model identity, predicted level, probabilities and
input-quality notes. Rejected rows have no level or confidence score.

## Results

Reads the active bundle's held-out confusion matrix and derives per-class
precision, recall, F1 and support. Saved accuracy is for supplied/recovered
clinical concepts, not an end-to-end test of live Ollama translation.

## Cluster Analysis

Uses the same active SapBERT encoder to inspect complaint similarity and cluster
geometry in the full 768-dimensional space. These are representation diagnostics,
not classifier accuracy or clinical validation.

See [the GUI guide](docs/SapBERT_GUI.md) and
[the current comparison report](reports/triage_four_level_round4/).
