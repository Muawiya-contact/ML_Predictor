"""Fixed full-dimensional versus PCA-64 comparison on the existing grouped split.

This descriptive follow-up does not tune or reselect the deployed model. Its
held-out rows have already appeared in the earlier report, so it is not a new
independent validation cohort. Run only after confirming the desired encoder.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse
import json
import time
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import research_engine as engine
from cache_integrity import file_sha256

def run(source, output, encoder):
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a new empty output directory; previous results are preserved.')
    df = pd.read_csv(source / 'dataset_with_splits.csv')
    array = source / f'emb_{encoder}.npy'
    meta = json.loads((source / f'emb_{encoder}.json').read_text())
    vectors = np.load(array, mmap_mode='r', allow_pickle=False)
    texts = df[meta['text_column']].fillna('').astype(str).tolist()
    import hashlib
    assert meta['text_sha256'] == hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    assert vectors.shape == (len(df), 768), 'This comparison requires genuine 768-D embeddings.'
    assert np.array_equal(df.row_id.to_numpy(), np.arange(len(df)))
    tr = np.flatnonzero(df.partition.eq('development'))
    te = np.flatnonzero(df.partition.eq('test'))
    assert set(df.iloc[tr].group).isdisjoint(df.iloc[te].group)
    y = df.Labels.to_numpy(dtype=int)
    assert set(y[tr]) == set(y[te]) == {1, 2, 3}
    params = {'logreg': {'C': 1, 'balance': True}, 'hgb': {'max_iter': 300, 'learning_rate': 0.1}, 'rf': {'n_estimators': 300, 'min_samples_leaf': 3, 'max_features': 0.7, 'balance': True}}
    plan = [engine.config(c, {'view': v, 'encoder': encoder, 'pca': p, 'solver': 'full'}, **params[c]) for v in ['text', 'fused'] for p in [0, 64] for c in params]
    (output / 'comparison_plan.json').write_text(json.dumps({'purpose': 'Fixed 2 x 2 x 3 descriptive comparison; no test-based selection', 'source_data_sha256': file_sha256(source / 'dataset_with_splits.csv'), 'embedding_sha256': file_sha256(array), 'encoder': meta, 'development_rows': len(tr), 'test_rows': len(te), 'configs': plan}, indent=2))
    engine.HERE = source
    results = []
    for view in ['text', 'fused']:
        for pca in [0, 64]:
            features = {'view': view, 'encoder': encoder, 'pca': pca, 'solver': 'full'}
            transform = engine.Features(**features).fit(df.iloc[tr])
            a = transform.transform(df.iloc[tr])
            b = transform.transform(df.iloc[te])
            for clf in params:
                config = engine.config(clf, features, **params[clf])
                name = f'{view}_{pca or 768}_{clf}'
                print('FIT', name, flush=True)
                start = time.monotonic()
                model = engine.fit_model(config, a, y[tr])
                pred = model.predict(b)
                result = {'id': name, 'config': config, 'feature_count': a.shape[1], 'fit_predict_seconds': time.monotonic() - start, 'metrics': engine.metrics(y[te], pred), 'report': classification_report(y[te], pred, labels=[1, 2, 3], output_dict=True, zero_division=0), 'confusion': confusion_matrix(y[te], pred, labels=[1, 2, 3]).tolist(), 'pca_retained_variance': float(transform.pca_.explained_variance_ratio_.sum()) if pca else None}
                pd.DataFrame({'row_id': te, 'reference': y[te], 'predicted': pred}).to_csv(output / f'{name}_predictions.csv', index=False)
                results.append(result)
                (output / 'results.json').write_text(json.dumps(results, indent=2))
                pd.DataFrame([{'id': r['id'], 'feature_count': r['feature_count'], **r['metrics']} for r in results]).to_csv(output / 'metrics.csv', index=False)
                print(name, json.dumps(result['metrics']), flush=True)
    assert file_sha256(array) == json.loads((output / 'comparison_plan.json').read_text())['embedding_sha256']
    print('COMPLETE: all twelve fixed comparisons', flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--encoder', choices=['sapbert_concept', 'mpnet_concept'], required=True)
    args = parser.parse_args()
    run(args.source, args.output, args.encoder)
