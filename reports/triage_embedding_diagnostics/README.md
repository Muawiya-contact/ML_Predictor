# Descriptive SapBERT geometry by triage label

A fixed stratified sample contains 1,200 development records, one per complaint
group. No test rows participate. Full-768 embeddings and the existing PCA-64/128
projections show weak cosine separation by supplied triage level; the two-PC
plot shows overlapping levels. This does not establish label errors or a
prediction ceiling. The deployed classifier also uses structured patient
measurements and original-complaint details.

The PCA was fitted on all development rows; this is descriptive geometry,
not cross-validated predictive performance. Centring changes cosine baselines,
so absolute raw/PCA similarities should not be treated as directly interchangeable.
Source/embedding/PCA hashes and sample details are recorded in `diagnostics.json`.
