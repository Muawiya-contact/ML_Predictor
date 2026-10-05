# Ordered models, nonlinear features and probability combinations

This follow-up preserves the four supplied labels, original development/test
partitions, complaint groups and the original emergency-recall constraint.
It does not alter the test set to obtain a target score. The existing test
records have already been examined; subsequent results are retrospective.

The development-only stages are:

1. Thirteen fixed-weight blends of existing out-of-fold probabilities plus the
   incumbent control. Every component used the same five grouped folds.
2. Twenty-four newly trained settings: cumulative ordinal logistic classifiers,
   cubic numeric feature expansions, and small supervised MLP classifiers.
   The MLP trains on frozen features; it does not fine-tune SapBERT.
3. Eleven additional blend settings: eight LR/HGB weight choices and three
   combinations with the ordered classifier. These were declared before their
   scores were calculated. Weights are fixed settings, not a fitted meta-model.
4. Six LR settings using frozen SapBERT on the supplied concept plus the complete
   original complaint, separated by `[SEP]`, with a 128-token limit. PCA and all
   patient/detail transformations still fit only on each training fold.

All settings and unsuccessful results are retained. The ensemble refinement
reuses development evidence, so estimates remain subject to repeated selection.
Full encoder fine-tuning is a separate, more expensive experiment; it is not
claimed here. This is a finite comparison of promising methods, not an exhaustive
search of every possible algorithm or hyperparameter.

```bash
python experiments/triage_study/probability_blends.py --root output/results \
  --output output/results/triage_probability_blends
python experiments/triage_study/advanced_classifiers.py \
  --original output/results/triage_four_level \
  --output output/results/triage_advanced_classifiers
python experiments/triage_study/verify_detail_refinement.py \
  --source output/results/triage_advanced_classifiers \
  --original output/results/triage_four_level \
  --incumbent output/results/triage_four_level_round6
python experiments/triage_study/probability_blends.py --root output/results \
  --output output/results/triage_refined_blends --refine
python experiments/triage_study/encode_complaint_pairs.py
python experiments/triage_study/paired_classifiers.py \
  --original output/results/triage_four_level \
  --output output/results/triage_paired_classifiers
python experiments/triage_study/finalize_advanced.py --root output/results \
  --output output/results/triage_four_level_round7
python experiments/triage_study/compare_tuned_families.py \
  --source output/results/triage_four_level_round7 \
  --original output/results/triage_four_level \
  --reuse output/results/triage_four_level_round6
python experiments/triage_study/verify_improvement.py \
  --source output/results/triage_four_level_round7 \
  --original output/results/triage_four_level
```

Use fresh destinations. The joint table contains 133 configurations: 109
trainable settings (545 five-fold fits across all rounds) and 24 blends (120
fold-wise probability evaluations, without new component fits). Do not label
all 665 evaluations as training runs. Deployment is selected before its final
retrospective evaluation, using macro F1 and the unchanged emergency-recall rule.

`fit_soft_vote.py` fits components on development rows and checks that their
feature columns can be mapped exactly into a shared feature matrix, on both
development and test inputs. Only then is the compact ensemble exported. The
shared adapter must reproduce every saved probability before promotion.

The 90% objective concerns accuracy and macro precision, recall and F1. Class
scores are reported separately; an aggregate above 90% does not imply every
class score exceeds 90%, or establish clinical validity. Source records,
individual predictions and OOF arrays remain local.

Export to a fresh staging directory and verify before promotion:

```bash
python experiments/triage_study/export_improved_bundle.py \
  --source output/results/triage_four_level_round7 \
  --original output/results/triage_four_level \
  --incumbent output/results/four_level_before_round7 \
  --output output/results/four_level_round7_bundle
python experiments/triage_study/verify_improved_serving.py \
  --source output/results/triage_four_level_round7 \
  --original output/results/triage_four_level \
  --bundle output/results/four_level_round7_bundle
```

Keep the previous serving directory as `four_level_before_round7` before replacing
it. The selected paired-text adapter requires the original complaint and the
checked English translation, matching the study's two-text input contract.
