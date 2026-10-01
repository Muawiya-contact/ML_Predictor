# Article methodology: current four-level study

The current study uses the supplied 10,000-row four-level cardiac workbook.
It compares Logistic Regression, Hist Gradient Boosting and Random Forest with
frozen SapBERT embeddings. The application uses the fused PCA-64 configuration
selected by five-fold grouped development cross-validation; the manifest records
the winning classifier and parameters.

1. Validate labels 0 Emergency, 1 Urgent, 2 Standard and 3 Non-urgent against the
   workbook's class names. Preserve the new targets throughout processing.
2. Recover the 4,290 missing Clinical_Concept values only where all eleven
   complaint/patient inputs and populated concepts match the previous supplied
   file. Exclude target names and processing metadata from input features.
3. Freeze approximately 80/20 development/holdout partitions, keeping connected
   groups of repeated complaints or concepts together. Use five grouped folds
   inside development for model selection.
4. Encode clinical concepts with the pinned SapBERT checkpoint, CLS pooling,
   64-token truncation and L2 normalization to obtain 768-dimensional vectors.
   The encoder remains frozen; this is classifier retraining, not BERT fine-tuning.
5. Fit PCA on training embeddings to reduce 768 dimensions to 64. Fit numeric
   imputation/scaling, ordinal AVPU and categorical encoding on training rows
   within each fold. Fusion contributes 22 structured features.
6. Evaluate all 12 fixed conditions: three classifiers × full/PCA embeddings ×
   text-only/fused inputs. Keep settings and holdout rows identical across views.
7. Compare ten fused PCA-64 configurations by development mean macro F1, with
   emergency recall and lower under-triage as tie breakers. Freeze selection
   before evaluating the holdout. Export that fitted model without refitting.
8. Report accuracy, macro precision/recall/F1, class-level scores and four-class
   confusion matrices, under-/over-triage and group-bootstrap uncertainty.
9. Describe the GUI's local Ollama translation, anatomical gate and six tabs
   separately from the classifier experiment. The GUI and report share the
   selected classifier, while the experiment uses supplied/recovered concepts.

Use values from [the current report](reports/triage_four_level/) and
`triage_model_sapbert/triage_metrics.json`. Previous three-level and MiniLM results
are separate experiments and are not interchangeable with these values.

The provider described label assignment as “by using all”; exact methods and
independent per-record review are not documented. Do not infer clinician review
or clinical validation. The study measures agreement with workbook labels;
live translation performance and clinical effectiveness require separate study.
The literature comparison uses a different binary KTAS dataset/task and is
included as context, not a direct superiority benchmark.
