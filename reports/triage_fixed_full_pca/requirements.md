# Requirements extracted from the supplied video

Reviewed the 2:07 recording using sampled frames and a local multilingual speech transcript. Technical terms were cross-checked against the displayed article; unclear literal transcription fragments were not treated as requirements.

1. **00:00-00:22:** Two comparison tables: text-only and text-plus-patient features. Each must contain all three classifiers at full 768-D and PCA-64, with accuracy, precision and F1.
2. **00:22-00:32:** Corresponding performance charts. Classifiers are the comparison categories; full 768-D and PCA-64 are paired series. Include accuracy and precision.
3. **00:54-01:20:** For Logistic Regression, show four confusion matrices: text-only/full, text-only/PCA-64, combined/full and combined/PCA-64.
4. **01:20-02:07:** Repeat the same complete confusion-matrix comparison for Hist Gradient Boosting and Random Forest. The spoken summary reiterates all three classifiers, both dimensions and both input views.
5. **User clarification:** Use SapBERT from the latest uploaded triage_classifier.py, on the latest 10,000-record dataset.
6. **Earlier retained requirement:** Include the screenshot-style Model / Accuracy / Recall / F1 literature-comparison table, with verified sources and clear task differences.

Implementation: twelve fixed experiments (3 classifiers x 2 dimensions x 2 input views). One encoder, identical grouped test rows and fixed per-classifier parameters across the four input conditions. The new dataset contains three target levels, so each matrix is 3x3 even though the older article examples were 4x4.
