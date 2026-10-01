# Current-data error investigation

These aggregate diagnostics describe the pre-refinement C=100 model on 8,001
development rows. Of 881 errors, 878 were between adjacent levels, 354 involved
Urgent versus Standard, and 220 had confidence at least 90%. All three classifier
families agreed on an incorrect level for 345 rows. These patterns do not prove
that reference labels are clinically incorrect; no labels were changed.

Thirteen predeclared settings were evaluated on all five immutable grouped folds.
`refinement/verification.json` checks all 65 fits, source hashes, group separation
and OOF metrics. Selection and diagnostics never use test rows. The best eligible
setting is balanced LR C=10, PCA-128 with the same quadratic patient and nineteen
original-complaint features. Its small gain over C=100 is uncertain.

The coefficient diagnostics are descriptive associations, not causal effects.
Individual error queues and raw records are excluded. Current model scores and
the single final PDF are in [the current report](../triage_four_level_round5/).
