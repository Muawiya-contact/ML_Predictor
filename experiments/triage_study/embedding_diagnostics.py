"""Descriptive triage-label geometry; not a classifier accuracy estimate."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cache_integrity import file_sha256


def run(original, bundle, output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a fresh diagnostic destination')
    df = pd.read_csv(original / 'dataset_with_splits.csv')
    dev = df[df.partition.eq('development')].drop_duplicates('group')
    indices, _ = train_test_split(np.arange(len(dev)), train_size=1200,
                                 stratify=dev.Labels, random_state=20261004)
    sample = dev.iloc[indices]
    assert sample.partition.eq('development').all() and sample.group.is_unique
    embedding = np.load(original / 'emb_sapbert_concept.npy', mmap_mode='r')
    raw = embedding[sample.row_id]
    manifest = json.loads((bundle / 'model_manifest.json').read_text())
    if file_sha256(bundle / 'pca.pkl') != manifest['artifact_sha256']['pca.pkl']:
        raise ValueError('PCA artifact hash mismatch')
    pca = joblib.load(bundle / 'pca.pkl')
    if pca.n_components_ < 128:
        raise ValueError('This diagnostic requires the evaluated PCA-128 or larger projection')
    projected = pca.transform(raw)
    labels = sample.Labels.to_numpy()
    same = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones(same.shape, dtype=bool), k=1)
    metrics = []
    for name, x in [('SapBERT 768-D', raw), ('PCA-64', projected[:, :64]), ('PCA-128', projected[:, :128])]:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        unit = x / norms.clip(1e-12)
        similarity = np.clip(unit @ unit.T, -1, 1)
        distance = np.clip(1 - similarity, 0, 2)
        np.fill_diagonal(distance, 0)
        within = float(similarity[upper & same].mean())
        between = float(similarity[upper & ~same].mean())
        metrics.append(dict(representation=name, within_label_cosine=within,
                            between_label_cosine=between, separation=within-between,
                            cosine_silhouette=float(silhouette_score(distance, labels, metric='precomputed'))))
    pd.DataFrame(metrics).to_csv(output / 'embedding_geometry.csv', index=False)
    result = dict(sample_rows=len(sample), sample_groups=int(sample.group.nunique()),
                  sample_class_counts=sample.Labels.value_counts().sort_index().to_dict(),
                  sample_seed=20261004, test_rows_used=False,
                  dataset_sha256=file_sha256(original / 'dataset_with_splits.csv'),
                  embedding_sha256=file_sha256(original / 'emb_sapbert_concept.npy'),
                  pca_sha256=file_sha256(bundle / 'pca.pkl'),
                  interpretation='Descriptive triage-label geometry, not cross-validated prediction. PCA was fitted on all development rows. One row per sampled complaint group. Low separation does not establish a performance ceiling or incorrect labels.',
                  metrics=metrics)
    (output / 'diagnostics.json').write_text(json.dumps(result, indent=2))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for level, color, name in zip(range(4), ['#c83f45','#e9a51d','#2585ab','#4f9d69'], ['Emergency','Urgent','Standard','Non-urgent']):
        mask = labels == level
        ax.scatter(projected[mask, 0], projected[mask, 1], c=color, s=13,
                   alpha=.45, label=f'{level} {name}')
    ax.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% variance)')
    ax.set_title('SapBERT concepts coloured by supplied triage level\n1,200 development records; one per complaint group')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output / 'triage_embedding_projection.png', dpi=190)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('original','bundle','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    a = parser.parse_args()
    run(a.original,a.bundle,a.output)
