# Current four-level SapBERT comparison

[Final report](SapBERT_Final_Report.pdf): 18 pages, retaining the original full-768/PCA-64 baseline comparisons, all 133 development settings, nine tuned families, confusion matrices, class scores, error/learning audits and paired-input embedding geometry. All pages were rendered and visually inspected.

The selected model is frozen SapBERT on `Clinical_Concept + " [SEP] " + chief_complaint`, with 128-token truncation, PCA-64, 50 patient features and 19 complaint-detail features (133 total), then balanced Logistic Regression C=100. At inference, the checked local Ollama translation replaces the supplied concept. The original complaint is retained verbatim. No encoder fine-tuning, label changes or partition changes occurred.

| Metric | Mean grouped development CV | Retrospective test |
| --- | ---: | ---: |
| Accuracy | 90.49% | 90.60% |
| Macro precision | 90.75% | 90.98% |
| Macro recall | 90.97% | 91.00% |
| Macro F1 | 90.83% | 90.99% |

All four aggregate targets exceed 90%; individual Urgent and Standard scores remain below 90%. Emergency recall changes from 95.10% to 94.36%, while overall undertriage decreases from 4.65% to 4.20%. This is a tradeoff, not an improvement in every metric. Conditional paired-group bootstrap development F1 gain versus the previous model is +1.18 percentage points (95% interval +0.70 to +1.67); this excludes repeated-selection uncertainty.

The joint comparison includes 109 trainable configurations (545 five-fold fits) and 24 fixed probability blends (120 fold evaluations reusing OOF predictions): 665 evaluation rows, not 665 training fits. This is a finite search of promising alternatives, not an exhaustive search. Family representatives may use different feature pipelines; the PDF identifies their settings.

Selection uses development macro F1 and the original emergency-recall constraint. The 1,999 test rows have previously been examined, so their results are retrospective, not fresh independent confirmation. Live Ollama translation accuracy and clinical validity are not established by these scores. Source label assignment and independent review remain undocumented.

Verification: all 665 CV rows and nine family results recomputed; source hashes and group separation checked; all 1,999 selected predictions and probabilities reproduced by the serving adapter; 13 live embedding checks and one live English prediction passed. All 49 targeted unit tests and the real six-tab GUI audit passed. The verified bundle is deployed locally; `deployment_decision.json` records its hashes and target assessment.

[Reproduction protocol](../../experiments/triage_study/ADVANCED_SEARCH.md). Only aggregate evidence is published: records, OOF arrays, individual predictions and development error queues remain local. Earlier report directories are historical snapshots.
