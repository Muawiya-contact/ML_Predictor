"""Reproducible 768-D versus PCA-64 triage experiment; never modifies serving models.

Input NPZ must contain numeric `embeddings` (N, 768) and unique `row_ids` (N,).
CSV rows are matched by an explicit ID, not by an assumed positional ordering.
Frozen embeddings must be generated without using evaluation labels. All learned
preprocessing below, including PCA, is fitted exclusively on training rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import time

# Set OpenMP defaults before importing compiled estimators.
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

NUMERIC = ['Age', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP', 'Temperature', 'SpO2']
CATEGORICAL = ['Gender', 'Mode_of_Arrival', 'AVPU', 'ECG_Status']


def load_inputs(csv_path, embeddings_path, id_column, target):
    """Validate dimensionality and join embeddings to labelled records by ID."""
    frame = pd.read_csv(csv_path, dtype={id_column: str})
    if id_column not in frame or target not in frame:
        raise ValueError(f'CSV requires {id_column!r} and {target!r} columns.')
    ids = frame[id_column]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError('CSV IDs must be nonmissing and unique.')
    with np.load(embeddings_path, allow_pickle=False) as data:
        vectors = data['embeddings']
        embedding_ids = data['row_ids'].astype(str)
    if vectors.ndim != 2 or vectors.shape[1] != 768:
        raise ValueError(f'Expected genuine 768-D embeddings; received {vectors.shape}.')
    if not np.issubdtype(vectors.dtype, np.number) or not np.isfinite(vectors).all():
        raise ValueError('Embeddings must contain finite numeric values.')
    if embedding_ids.ndim != 1 or len(embedding_ids) != len(vectors):
        raise ValueError('One row ID is required per embedding.')
    if len(set(embedding_ids)) != len(embedding_ids) or set(ids) != set(embedding_ids):
        raise ValueError('CSV and embedding IDs must match exactly and be unique.')
    positions = {value: i for i, value in enumerate(embedding_ids)}
    vectors = vectors[[positions[value] for value in ids]].astype(np.float64)
    if frame[target].isna().any():
        raise ValueError('Target labels must not be missing.')
    labels = pd.to_numeric(frame[target], errors='raise').to_numpy()
    if not np.isin(labels, [1, 2, 3, 4]).all() or len(np.unique(labels)) < 2:
        raise ValueError('This experiment requires triage levels 1–4 and at least two classes.')
    return frame, vectors, labels.astype(int)


def representations(vectors, train, test, components=64):
    """Fit PCA on training embeddings only; reject impossible reductions."""
    if min(len(train) - 1, vectors.shape[1]) < components:
        raise ValueError(f'PCA-{components} requires at least {components + 1} training rows.')
    pca = PCA(n_components=components, svd_solver='full')
    reduced_train = pca.fit_transform(vectors[train])
    reduced_test = pca.transform(vectors[test])
    return {'full_768': (vectors[train], vectors[test]),
            'pca_64': (reduced_train, reduced_test)}, pca


def structured_features(frame, train, test):
    """Imputation, numeric scaling and one-hot vocabulary use training rows only."""
    missing = set(NUMERIC + CATEGORICAL) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing structured columns: {sorted(missing)}')
    frame = frame.copy()
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    for column in CATEGORICAL:
        frame[column] = frame[column].map(lambda x: str(x) if pd.notna(x) else np.nan)
    transform = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                                  StandardScaler()), NUMERIC),
        ('categorical', make_pipeline(SimpleImputer(strategy='most_frequent', keep_empty_features=True),
                                      OneHotEncoder(handle_unknown='ignore', sparse_output=False)), CATEGORICAL)
    ], sparse_threshold=0)
    return transform.fit_transform(frame.iloc[train]), transform.transform(frame.iloc[test])


def models(seed):
    return {
        'LogisticRegression': LogisticRegression(max_iter=3000, class_weight='balanced', random_state=seed),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, early_stopping=False,
                                                              random_state=seed),
        'RandomForest': RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                               n_jobs=1, random_state=seed),
    }


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    frame, vectors, labels = load_inputs(args.data, args.embeddings, args.id_column, args.target)
    train, test = train_test_split(np.arange(len(labels)), test_size=args.test_size,
                                   random_state=args.seed, stratify=labels)
    blocks, pca = representations(vectors, train, test)
    if set(labels[train]) != set(labels) or set(labels[test]) != set(labels):
        raise ValueError('Every class must occur in both train and test; adjust data/split.')
    extra = structured_features(frame, train, test) if args.structured else None
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty; use a new run directory to preserve prior results.')
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'row_id': frame[args.id_column],
                  'partition': np.where(np.isin(np.arange(len(labels)), train), 'train', 'test')}).to_csv(
                      output / 'split.csv', index=False)
    np.savez_compressed(output / 'pca_parameters.npz', components=pca.components_, mean=pca.mean_,
                        explained_variance_ratio=pca.explained_variance_ratio_)
    rows, per_class, predictions = [], [], []
    all_labels = np.unique(labels)
    for representation, (xtrain, xtest) in blocks.items():
        if extra is not None:
            xtrain, xtest = np.hstack([xtrain, extra[0]]), np.hstack([xtest, extra[1]])
        for name, estimator in models(args.seed).items():
            print(f'Training {representation}: {name}', flush=True)
            started = time.perf_counter()
            estimator.fit(xtrain, labels[train])
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            predicted = estimator.predict(xtest)
            predict_seconds = time.perf_counter() - started
            report = classification_report(labels[test], predicted, labels=all_labels,
                                           output_dict=True, zero_division=0)
            rows.append(dict(representation=representation, classifier=name, input_features=xtrain.shape[1],
                             accuracy=accuracy_score(labels[test], predicted),
                             precision_macro=report['macro avg']['precision'],
                             precision_weighted=report['weighted avg']['precision'],
                             recall_macro=report['macro avg']['recall'], f1_macro=report['macro avg']['f1-score'],
                             under_triage_rate=float(np.mean(predicted > labels[test])),
                             over_triage_rate=float(np.mean(predicted < labels[test])),
                             fit_seconds=fit_seconds, predict_seconds=predict_seconds))
            for level in all_labels:
                per_class.append(dict(representation=representation, classifier=name, level=int(level),
                                      **report[str(level)]))
            predictions.extend(dict(representation=representation, classifier=name, row_id=frame.iloc[i][args.id_column],
                                    true_level=int(actual), predicted_level=int(pred))
                               for i, actual, pred in zip(test, labels[test], predicted))
            matrix = confusion_matrix(labels[test], predicted, labels=all_labels)
            stem = f'{representation}_{name}'
            pd.DataFrame(matrix, index=all_labels, columns=all_labels).to_csv(output / f'{stem}_confusion.csv')
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(matrix, display_labels=all_labels).plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_title(f'{name} · {representation}')
            fig.tight_layout(); fig.savefig(output / f'{stem}_confusion.png', dpi=180); plt.close(fig)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / 'comparison_metrics.csv', index=False)
    pd.DataFrame(per_class).to_csv(output / 'per_class_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(output / 'predictions.csv', index=False)
    for metric in ['accuracy', 'precision_macro']:
        fig, ax = plt.subplots(figsize=(9, 5))
        metrics.pivot(index='classifier', columns='representation', values=metric).plot.bar(ax=ax, rot=0)
        ax.set_ylim(0, 1); ax.set_ylabel(metric.replace('_', ' ').capitalize())
        ax.set_title('Identical held-out rows: full embeddings versus PCA-64')
        fig.tight_layout(); fig.savefig(output / f'{metric}_comparison.png', dpi=180); plt.close(fig)
    columns = ['representation', 'classifier', 'accuracy', 'precision_macro', 'f1_macro']
    table = '| ' + ' | '.join(columns) + ' |\n|' + '|'.join(['---'] * len(columns)) + '|\n'
    for _, row in metrics.iterrows():
        table += '| ' + ' | '.join(f'{row[c]:.4f}' if isinstance(row[c], float) else str(row[c]) for c in columns) + ' |\n'
    (output / 'comparison_table.md').write_text(table)
    (output / 'comparison_table.tex').write_text(metrics[columns].to_latex(index=False, escape=True, float_format='%.4f'))
    metadata = dict(seed=args.seed, test_size=args.test_size, train_rows=len(train), test_rows=len(test),
                    embedding_model=args.embedding_model, structured=args.structured, target=args.target,
                    data_sha256=fingerprint(args.data), embeddings_sha256=fingerprint(args.embeddings),
                    pca_variance_retained=float(pca.explained_variance_ratio_.sum()),
                    sklearn=sklearn.__version__, numpy=np.__version__, python=platform.python_version(),
                    model_parameters={name: m.get_params() for name, m in models(args.seed).items()},
                    protocol='Single stratified held-out split; train-only PCA and structured preprocessing; no tuning.')
    (output / 'run_manifest.json').write_text(json.dumps(metadata, indent=2))
    print(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--embeddings', required=True, type=Path)
    parser.add_argument('--embedding-model', required=True, help='Exact checkpoint/revision and pooling used for the input vectors')
    parser.add_argument('--id-column', default='row_id')
    parser.add_argument('--target', default='Triage_Level')
    parser.add_argument('--structured', action='store_true', help='Append train-preprocessed patient features after embedding/PCA')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--output', default='output/results/bert_comparison')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args)


if __name__ == '__main__':
    main()
