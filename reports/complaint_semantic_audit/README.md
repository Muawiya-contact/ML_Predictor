# Current paired-SapBERT complaint-mention diagnostic

Four seeds (42, 99, 404, 777), each with 100 distinct development complaint groups:
20 each mentioning arm, back, jaw, palpitations or shoulder. Literal word rules
are applied to the supplied English concept; zero/multiple matches are excluded.
The eligible pool has 1,915 groups. The four runs use 368 distinct records overall;
samples may overlap. No test rows or new encoder/classifier training are used.

These are **rule-derived mentions**, not reviewed diagnoses or independent
semantic ground truth. All plotted colours are labels, not KMeans output.

- Figure 1: complaint mentions on the first two deployed PCA components.
- Figure 2: exactly the same samples/coordinates coloured by supplied urgency.
- Figure 3: those samples in a separate label-blind PCA fitted once to all eligible
  development vectors. This diagnostic projection is not deployed.

`metrics.csv` retains every run and space: distances, silhouette, nearest-mention
matching, KMeans ARI and ANOSIM. ANOSIM uses 499 permutations and Holm correction
across all 16 tests. KMeans has five clusters and 20 initializations. None of these
scores is triage classification accuracy. The derivation of groups from the input
text and overlap between samples limit interpretation of significance tests.

Sample sizes, unique groups, development-only membership, artifact hashes and
full-dimensional distance/silhouette/nearest-neighbour calculations were checked.
Independent distance recalculation used SciPy `cdist`. Original classifier
artifacts are unchanged. Private sample records and row identifiers are excluded.

See [the general workflow PDF](../triage_selected_model/Roman_Urdu_Triage_Project_Workflow.pdf)
for plain-language interpretation. Reproduce with
`python experiments/triage_study/complaint_semantic_audit.py` from the repository
root after generating the original paired embeddings and fitted serving bundle.
