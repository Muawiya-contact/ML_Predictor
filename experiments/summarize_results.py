"""Build a reader-facing results index from completed comparison runs."""
import json
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1] / 'output/results'
    metrics = pd.read_csv(root / 'all_comparison_metrics.csv')
    inputs = json.loads((root / 'sbert_inputs/embedding_manifest.json').read_text())
    text_split = pd.read_csv(root / 'sbert_text_only/split.csv')
    fusion_split = pd.read_csv(root / 'sbert_fusion/split.csv')
    pd.testing.assert_frame_equal(text_split, fusion_split)
    report = ['# Sections 3.4 and 3.5: classifier results', '',
              f"Encoder: `{inputs['model']}` at commit `{inputs['revision']}`.",
              'Frozen, normalized 768-dimensional embeddings of existing English translations.',
              'Data: 2,252 synthetic cardiac records; 1,801 training and 451 test rows; seed 42.',
              'All runs use identical test rows. PCA and structured preprocessing use training rows only.', '',
              '| Features | Representation | Classifier | Accuracy | Macro precision | Macro F1 |',
              '|---|---|---|---:|---:|---:|']
    for row in metrics.itertuples():
        report.append(f'| {row.feature_set} | {row.representation} | {row.classifier} | '
                      f'{row.accuracy:.2%} | {row.precision_macro:.2%} | {row.f1_macro:.2%} |')
    report += ['', '## Files', '',
               '- `section_3_4_metrics.csv` / `section_3_4_table.tex`: original 768-D comparison.',
               '- `section_3_5_metrics.csv` / `section_3_5_table.tex`: PCA-64 comparison.',
               '- `all_comparison_metrics.csv`: all twelve runs, including fusion ablation and timings.',
               '- `sbert_text_only/` and `sbert_fusion/`: accuracy/precision graphs, confusion matrices, per-class scores and predictions.',
               '- `sbert_inputs/embedding_manifest.json`: exact checkpoint and input provenance.', '',
               'Fusion appends patient features, so its total classifier input is larger than 768 or 64.',
               'CSV scores are fractions; the table above displays percentages. These are single-split results, not cross-validation means.',
               'The dataset is synthetic, and repeated complaint phrases may cross the random row split.',
               'The results do not establish clinical safety or measure the live translation pipeline.', '',
               '## Section 3.6 — pending source requirements', '',
               'The article contains only “Comparing with State-of-the-Art”; no notebook or published comparator list was supplied.',
               '`section_3_6_comparison_template.csv` records the required comparison fields. No literature scores have been invented.', '',
               'Reproduction instructions: [experiments/README.md](../../experiments/README.md).']
    (root / 'README.md').write_text('\n'.join(report) + '\n')
    print(metrics[['feature_set', 'representation', 'classifier', 'accuracy', 'precision_macro']].to_string(index=False))


if __name__ == '__main__':
    main()
