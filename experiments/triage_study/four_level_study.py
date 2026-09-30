"""Train a fixed, leakage-controlled four-level SapBERT comparison.

Five-fold grouped development CV selects a fused PCA-64 classifier using
macro F1, then emergency recall, then lower under-triage as tie breakers.
The fixed 12 baseline holdout conditions are also reported in full.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse
import hashlib
import json
import shutil
import time
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import research_engine as engine
import prepare_data
from cache_integrity import file_sha256
from prepare_four_level_source import PROVENANCE

LABELS = [0, 1, 2, 3]
NAMES = ['Emergency', 'Urgent', 'Standard', 'Non-urgent']
PARAMS = {'logreg': {'C': 1, 'balance': True},
          'hgb': {'max_iter': 300, 'learning_rate': .1},
          'rf': {'n_estimators': 300, 'min_samples_leaf': 3, 'max_features': .7, 'balance': True}}


def scores(y, pred):
    report = classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0)
    return {'accuracy': float(np.mean(y == pred)), 'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'], 'macro_f1': report['macro avg']['f1-score'],
            'qwk': cohen_kappa_score(y, pred, labels=LABELS, weights='quadratic'),
            'mae': float(np.abs(y - pred).mean()), 'under_triage_rate': float((pred > y).mean()),
            'over_triage_rate': float((pred < y).mean()),
            'emergency_recall': report['0']['recall']}


def run(source, output, embedding_source=None, model_path=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a fresh destination; no partial or stale results are reused.')
    prepare_data.HERE = output
    prepare_data.main(source, target_column='Triage_Level', labels=LABELS,
                      provenance=PROVENANCE)
    df = pd.read_csv(output / 'dataset_with_splits.csv')
    texts = df.Clinical_Concept.fillna('').astype(str).tolist()
    fingerprint = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    if embedding_source is None:
        # Frozen encoding is label-independent. Generate the exact same vectors
        # from a local snapshot when no verified cache was supplied.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.sapbert_serving import SapBERTEncoder
        settings = {'revision': '090663c3ae57bf35ffe4d0d468a2a88d03051a4d',
                    'pooling': 'cls', 'max_token_length': 64, 'normalized': True,
                    'local_path': str(model_path or 'local_sapbert_snapshot')}
        encoder = SapBERTEncoder({'encoder_settings': settings,
            'embedding_model': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'})
        unique = list(dict.fromkeys(texts))
        vectors = encoder.encode(unique)
        lookup = {text: i for i, text in enumerate(unique)}
        full = vectors[[lookup[text] for text in texts]]
        np.save(output / 'emb_sapbert_concept.npy', full)
        meta = {'name': 'sapbert_concept', 'text_column': 'Clinical_Concept',
                'model_path': settings.pop('local_path'), **settings,
                'text_sha256': fingerprint, 'shape': list(full.shape),
                'array_sha256': file_sha256(output / 'emb_sapbert_concept.npy')}
        (output / 'emb_sapbert_concept.json').write_text(json.dumps(meta, indent=2))
        embedding_source = output
    meta = json.loads((embedding_source / 'emb_sapbert_concept.json').read_text())
    if fingerprint != meta['text_sha256'] or meta['revision'] != '090663c3ae57bf35ffe4d0d468a2a88d03051a4d':
        raise ValueError('Concept text or checkpoint changed; generate fresh SapBERT embeddings for this output.')
    array = embedding_source / 'emb_sapbert_concept.npy'
    if meta.get('array_sha256') != file_sha256(array):
        raise ValueError('Embedding array checksum does not match its recorded identity.')
    if np.load(array, mmap_mode='r', allow_pickle=False).shape != (len(df), 768):
        raise ValueError('Embedding rows/dimensions do not match the new dataset.')
    if array.resolve() != (output / array.name).resolve():
        shutil.copy2(array, output / array.name)
    meta['array_sha256'] = file_sha256(output / array.name)
    (output / 'emb_sapbert_concept.json').write_text(json.dumps(meta, indent=2))
    source_meta = source.parent / 'source_metadata.json'
    if source_meta.exists():
        shutil.copy2(source_meta, output / source_meta.name)
    y = df.Labels.to_numpy(dtype=int)
    tr = np.flatnonzero(df.partition.eq('development'))
    te = np.flatnonzero(df.partition.eq('test'))
    assert set(y[tr]) == set(y[te]) == set(LABELS)
    assert set(df.iloc[tr].group).isdisjoint(df.iloc[te].group)
    engine.HERE = output
    plan = [engine.config(c, {'view': v, 'encoder': 'sapbert_concept', 'pca': p, 'solver': 'full'}, **PARAMS[c])
            for v in ['text', 'fused'] for p in [0, 64] for c in PARAMS]
    cv_candidates = {c + '_baseline': engine.config(c, {'view': 'fused', 'encoder': 'sapbert_concept', 'pca': 64, 'solver': 'full'}, **PARAMS[c]) for c in PARAMS}
    alternatives = [
        ('logreg_regularized', 'logreg', {'C': .1, 'balance': True}),
        ('logreg_c10', 'logreg', {'C': 10, 'balance': True}),
        ('hgb_regularized', 'hgb', {'max_iter': 300, 'max_leaf_nodes': 15, 'l2_regularization': 1, 'learning_rate': .1, 'balance': True}),
        ('hgb_slow', 'hgb', {'max_iter': 500, 'max_leaf_nodes': 15, 'l2_regularization': 1, 'learning_rate': .05}),
        ('hgb_balanced', 'hgb', {'max_iter': 300, 'max_leaf_nodes': 31, 'l2_regularization': 10, 'learning_rate': .1, 'balance': True}),
        ('rf_fine', 'rf', {'n_estimators': 400, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'balance': False}),
        ('rf_balanced', 'rf', {'n_estimators': 400, 'min_samples_leaf': 2, 'max_features': .5, 'balance': True}),
    ]
    for name, c, params in alternatives:
        cv_candidates[name] = engine.config(c, {'view': 'fused', 'encoder': 'sapbert_concept', 'pca': 64, 'solver': 'full'}, **params)
    manifest = {'purpose': 'Fixed four-level comparison with development-only cross-validation',
                'labels': LABELS, 'label_names': NAMES, 'serving_id': 'selected_fused_64',
                'cv_candidates': cv_candidates,
                'selection': 'Highest five-fold mean macro F1; ties use emergency recall then lower under-triage; holdout unused for selection',
                'source_data_sha256': file_sha256(output / 'dataset_with_splits.csv'),
                'embedding_sha256': meta['array_sha256'], 'encoder': meta,
                'development_rows': len(tr), 'test_rows': len(te), 'configs': plan}
    (output / 'comparison_plan.json').write_text(json.dumps(manifest, indent=2))
    cv = []
    for fold in range(5):
        fit = np.flatnonzero(df.partition.eq('development') & df.cv5.ne(fold))
        val = np.flatnonzero(df.partition.eq('development') & df.cv5.eq(fold))
        assert set(df.iloc[fit].group).isdisjoint(df.iloc[val].group)
        transform = engine.Features(view='fused', encoder='sapbert_concept', pca=64, solver='full').fit(df.iloc[fit])
        a, b = transform.transform(df.iloc[fit]), transform.transform(df.iloc[val])
        for candidate, config in cv_candidates.items():
            c = config['classifier']
            print('CV', fold + 1, candidate, flush=True)
            model = engine.fit_model(config, a, y[fit])
            pred = model.predict(b)
            cv.append({'fold': fold + 1, 'candidate': candidate, 'classifier': c, 'train_rows': len(fit), 'validation_rows': len(val), **scores(y[val], pred)})
            pd.DataFrame(cv).to_csv(output / 'cross_validation.csv', index=False)
    summary = pd.DataFrame(cv).groupby('candidate').agg(
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
        accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'),
        precision_mean=('precision_macro', 'mean'), recall_mean=('recall_macro', 'mean'),
        emergency_recall_mean=('emergency_recall', 'mean'),
        under_triage_mean=('under_triage_rate', 'mean')).reset_index()
    summary = summary.sort_values(['macro_f1_mean', 'emergency_recall_mean', 'under_triage_mean', 'candidate'], ascending=[False, False, True, True])
    summary.to_csv(output / 'cv_summary.csv', index=False)
    chosen = summary.iloc[0]['candidate']
    selection = {'candidate': chosen, 'config': cv_candidates[chosen], 'rule': manifest['selection']}
    (output / 'selection.json').write_text(json.dumps(selection, indent=2))
    print('FROZEN SELECTION', chosen, flush=True)
    results = []
    for view in ['text', 'fused']:
        for pc in [0, 64]:
            features = {'view': view, 'encoder': 'sapbert_concept', 'pca': pc, 'solver': 'full'}
            transform = engine.Features(**features).fit(df.iloc[tr])
            a, b = transform.transform(df.iloc[tr]), transform.transform(df.iloc[te])
            for c in PARAMS:
                name = f'{view}_{pc or 768}_{c}'
                config = engine.config(c, features, **PARAMS[c])
                print('FIT', name, flush=True)
                start = time.monotonic()
                model = engine.fit_model(config, a, y[tr])
                pred = model.predict(b)
                r = {'id': name, 'config': config, 'feature_count': a.shape[1],
                     'fit_predict_seconds': time.monotonic() - start, 'metrics': scores(y[te], pred),
                     'report': classification_report(y[te], pred, labels=LABELS, output_dict=True, zero_division=0),
                     'confusion': confusion_matrix(y[te], pred, labels=LABELS).tolist(),
                     'pca_retained_variance': float(transform.pca_.explained_variance_ratio_.sum()) if pc else None}
                pd.DataFrame({'row_id': te, 'reference': y[te], 'predicted': pred}).to_csv(output / f'{name}_predictions.csv', index=False)
                results.append(r)
                (output / 'results.json').write_text(json.dumps(results, indent=2))
                pd.DataFrame([{'id': x['id'], 'feature_count': x['feature_count'], **x['metrics']} for x in results]).to_csv(output / 'metrics.csv', index=False)
                print(name, r['metrics'], flush=True)
    config = selection['config']
    transform = engine.Features(**config['features']).fit(df.iloc[tr])
    a, b = transform.transform(df.iloc[tr]), transform.transform(df.iloc[te])
    model = engine.fit_model(config, a, y[tr])
    pred = model.predict(b)
    selected = {'id': 'selected_fused_64', 'candidate': chosen, 'config': config,
                'feature_count': a.shape[1], 'metrics': scores(y[te], pred),
                'report': classification_report(y[te], pred, labels=LABELS, output_dict=True, zero_division=0),
                'confusion': confusion_matrix(y[te], pred, labels=LABELS).tolist()}
    (output / 'selected_results.json').write_text(json.dumps(selected, indent=2))
    pd.DataFrame({'row_id': te, 'reference': y[te], 'predicted': pred}).to_csv(output / 'selected_fused_64_predictions.csv', index=False)
    dest = output / 'serving'; dest.mkdir(exist_ok=True)
    joblib.dump(model, dest / 'model.pkl')
    joblib.dump(transform.structured_, dest / 'structured.pkl')
    joblib.dump(transform.pca_, dest / 'pca.pkl')
    assert file_sha256(output / array.name) == manifest['embedding_sha256']
    assert file_sha256(output / 'dataset_with_splits.csv') == manifest['source_data_sha256']
    print('COMPLETE: 50 CV fits, all 12 baseline comparisons and selected model', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--embedding-source', type=Path, help='Optional verified cache; otherwise encode locally')
    parser.add_argument('--model-path', type=Path, help='Pinned local SapBERT snapshot for fresh encoding')
    a = parser.parse_args()
    run(a.source, a.output, a.embedding_source, a.model_path)
