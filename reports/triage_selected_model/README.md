# Current selected-model paper package

- [General project workflow](Roman_Urdu_Triage_Project_Workflow.pdf): eight pages explaining input, translation, embeddings, PCA, classification, cross-validation, weighting and the interpretation of complaint plots. No conversation-specific headings.
- [Selected-model results](SapBERT_Final_Paper_Report.pdf): eight pages with current metrics, class scores, confusion matrices, four triage-coloured embedding panels and repeated descriptive geometry. No historical baseline tables or alternative-classifier comparison.
- [Figure captions and usage](figures/Figure_Captions_and_Usage.txt): descriptive filenames, six main figures in PNG/vector PDF, four individual embedding panels, an embedding-statistics table image and aggregate supporting data.

Both reports concern the active frozen paired-text SapBERT + PCA-64 + balanced
Logistic Regression C=100 model (133 inputs). The measured 90.60% accuracy and
90.99% macro F1 are retrospective, not independent clinical validation. Individual
class scores do not all exceed 90%. No new classifier training was performed for
these documents.

Two different embedding analyses must not be confused:

1. The figure package here uses four balanced 200-record development samples,
   coloured by supplied **triage levels**. Its table summarizes 20 resamples.
2. The [complaint-mention diagnostic](../complaint_semantic_audit/) uses four
   balanced 100-record samples, coloured by literal arm/back/jaw/palpitations/
   shoulder mentions, and also shows the same points coloured by urgency.
   These mention labels are rule-derived, not independent clinical categories.

All predeclared runs are retained. A separate diagnostic PCA is clearly labelled
and is not deployed. Visual separation, nearest-neighbour matching and triage
accuracy are different measurements. Only aggregate evidence and figures are
published; patient text, identifiers and per-record predictions remain local.

Reproduce from the saved local experiment evidence, in repository root:

```bash
python experiments/triage_study/build_paper_embedding_figures.py
python experiments/triage_study/build_selected_model_report.py
python experiments/triage_study/complaint_semantic_audit.py
python experiments/triage_study/build_project_workflow_pdf.py
```

The original study caches under `output/results/triage_four_level` and round-seven
verified results are prerequisites. These contain local data and are deliberately
not included here. The PDF builders use ReportLab and embedded Vera fonts. All
16 report pages were rendered and visually checked. The original full search
remains in [the round-seven archive](../triage_four_level_round7/).
