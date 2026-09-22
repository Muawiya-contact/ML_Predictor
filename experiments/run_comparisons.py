"""Run text-only and structured-fusion ablations from prepared SBERT inputs."""
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    inputs = root / 'output/results/sbert_inputs'
    manifest = json.loads((inputs / 'embedding_manifest.json').read_text())
    model = manifest['model'] + '@' + manifest['revision'] + '; normalized mean pooling'
    for variant in ['text_only', 'fusion']:
        command = [sys.executable, str(root / 'experiments/compare_classifiers.py'),
                   '--data', str(inputs / 'labelled_rows.csv'),
                   '--embeddings', str(inputs / 'embeddings.npz'), '--embedding-model', model,
                   '--output', str(root / f'output/results/sbert_{variant}')]
        if variant == 'fusion':
            command.append('--structured')
        subprocess.run(command, check=True, cwd=root)
    tables = []
    for variant in ['text_only', 'fusion']:
        table = pd.read_csv(root / f'output/results/sbert_{variant}/comparison_metrics.csv')
        table.insert(0, 'feature_set', variant)
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(root / 'output/results/all_comparison_metrics.csv', index=False)
    for section, representation in [('3_4', 'full_768'), ('3_5', 'pca_64')]:
        subset = combined[combined.representation == representation]
        subset.to_csv(root / f'output/results/section_{section}_metrics.csv', index=False)
        subset[['feature_set', 'classifier', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro']].to_latex(
            root / f'output/results/section_{section}_table.tex', index=False, escape=True, float_format='%.4f')


if __name__ == '__main__':
    main()
