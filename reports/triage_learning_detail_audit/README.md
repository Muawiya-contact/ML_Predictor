# Development-only data and learning audit

Ninety fits reuse the five development folds without changing labels or
examining new test rows. Sixty learning-curve fits use nested whole-group
subsets; thirty additional fits compare concept and original-complaint detail
features. Full-size baselines reproduce the previous study exactly.

The audit flags 567 explicit duration disagreements, 142 durations not recovered
from the concept and 48 family-word omissions. Lexical flags require review and
do not establish clinical findings or incorrect labels. The local review queue
contains 1,782 distinct records; individual records and group memberships are
not published.

Original-complaint details improve mean LR macro F1 from 86.57% to 89.38%, while
meeting the original emergency-recall constraint. The combined final PDF in ../triage_four_level_round4/SapBERT_Final_Report.pdf contains
all nine full-size conditions and learning curves. These curves do not promise
a score at 20,000 rows. Independently reviewed new records are needed.

The combined model selection, retrospective evaluation and nine-page final
comparison are in ../triage_four_level_round4/.
