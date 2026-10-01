"""Grouped learning curves and predeclared complaint-detail experiments.

Only development records are used. All three classifier families, source labels,
SapBERT vectors and validation folds are fixed. More data is not extrapolated
into an invented 20,000-row performance estimate.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.complaint_details import detail_matrix, duration_minutes, FEATURE_NAMES, PATTERNS, VERSION
import research_engine as engine
from four_level_study import scores
from cache_integrity import file_sha256

FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def nested_group_subsets(frame, seed):
    """Nested whole-group samples; never fit preprocessing on excluded rows."""
    groups = np.random.default_rng(seed).permutation(frame.group.unique())
    sizes = frame.groupby('group').size().reindex(groups).to_numpy()
    cumulative = sizes.cumsum()
    subsets = {}
    for fraction in FRACTIONS:
        count = min(len(groups), int(np.searchsorted(cumulative, len(frame) * fraction)) + 1)
        subset = frame[frame.group.isin(groups[:count])].copy()
        if set(subset.Labels) != {0, 1, 2, 3}:
            raise ValueError('A training subset lacks a target level')
        subsets[fraction] = subset
    return subsets


def audit_text_and_errors(source, incumbent, output):
    all_rows = pd.read_csv(source / 'dataset_with_splits.csv')
    dev = all_rows[all_rows.partition.eq('development')].copy()
    oof = np.load(incumbent / 'development_oof.npz')
    np.testing.assert_array_equal(oof['row_id'], dev.row_id)
    np.testing.assert_array_equal(oof['reference'], dev.Labels)
    raw = detail_matrix(dev.chief_complaint.fillna('').tolist())
    concept = detail_matrix(dev.Clinical_Concept.fillna('').tolist())
    raw_d = [duration_minutes(t)[0] for t in dev.chief_complaint.fillna('')]
    concept_d = [duration_minutes(t)[0] for t in dev.Clinical_Concept.fillna('')]
    mismatch = np.array([a is not None and b is not None and a != b for a, b in zip(raw_d, concept_d)])
    missing = np.array([a is not None and b is None for a, b in zip(raw_d, concept_d)])
    family_index = FEATURE_NAMES.index('family_word')
    family_missing = (raw[:, family_index] == 1) & (concept[:, family_index] == 0)
    dev['oof_predicted'] = oof['selected'].argmax(1)
    dev['oof_probability'] = oof['selected'].max(1)
    dev['explicit_complaint_duration_minutes'] = raw_d
    dev['explicit_concept_duration_minutes'] = concept_d
    dev['duration_disagreement_flag'] = mismatch
    dev['duration_missing_from_concept_flag'] = missing
    dev['family_word_missing_from_concept_flag'] = family_missing
    dev['reviewed_label'] = ''
    dev['reviewer_reason'] = ''
    errors = dev.Labels.ne(dev.oof_predicted).to_numpy()
    review = dev[errors | mismatch | missing | family_missing].sort_values('oof_probability', ascending=False)
    review.to_csv(output / 'development_review_queue.csv', index=False)
    counts = {name: dict(complaint_mentions=int(raw[:, i].sum()), concept_mentions=int(concept[:, i].sum()),
                        complaint_only=int(((raw[:, i] == 1) & (concept[:, i] == 0)).sum()))
              for i, name in enumerate(FEATURE_NAMES) if name != 'log_duration_minutes'}
    audit = dict(development_rows=len(dev), oof_errors=int(errors.sum()), review_rows=len(review),
                 explicit_duration_disagreements=int(mismatch.sum()),
                 explicit_duration_missing_from_concept=int(missing.sum()),
                 family_word_missing_from_concept=int(family_missing.sum()),
                 lexical_screen=counts, labels_modified=False,
                 interpretation='Lexical flags require human review; they do not establish clinical findings or incorrect labels. No new test-row inspection.',
                 oof_confusion=pd.crosstab(dev.Labels, dev.oof_predicted).reindex(index=range(4), columns=range(4), fill_value=0).to_numpy().tolist())
    (output / 'input_error_audit.json').write_text(json.dumps(audit, indent=2))
    return audit


def evaluate_fold(source, fold, configs):
    engine.HERE = Path(source)
    frame = pd.read_csv(engine.HERE / 'dataset_with_splits.csv')
    dev = frame[frame.partition.eq('development')]
    train = dev[dev.cv5.ne(fold)]
    valid = dev[dev.cv5.eq(fold)]
    assert not set(train.group) & set(valid.group)
    records, predictions, membership = [], {}, []
    subsets = nested_group_subsets(train, 20261001 + fold)
    for fraction, subset in subsets.items():
        membership.append(dict(fold=fold + 1, fraction=fraction, train_row_ids=subset.row_id.tolist(),
                               validation_row_ids=valid.row_id.tolist(), train_groups=int(subset.group.nunique())))
        cache = {}
        for family, config in configs.items():
            feature_key = json.dumps(config['features'], sort_keys=True)
            if feature_key not in cache:
                features = engine.Features(**config['features']).fit(subset)
                cache[feature_key] = (features.transform(subset), features.transform(valid))
            a, b = cache[feature_key]
            variants = ['baseline'] if fraction != 1 else ['baseline', 'concept_details', 'complaint_details']
            for variant in variants:
                x, xv = a, b
                if variant != 'baseline':
                    column = 'Clinical_Concept' if variant == 'concept_details' else 'chief_complaint'
                    scaler = StandardScaler().fit(detail_matrix(subset[column].fillna('').tolist()))
                    x = np.hstack([a, scaler.transform(detail_matrix(subset[column].fillna('').tolist()))])
                    xv = np.hstack([b, scaler.transform(detail_matrix(valid[column].fillna('').tolist()))])
                started = time.monotonic()
                model = engine.fit_model(config, x, subset.Labels.to_numpy())
                np.testing.assert_array_equal(model.classes_, [0, 1, 2, 3])
                prob = model.predict_proba(xv)
                train_scores = scores(subset.Labels.to_numpy(), model.predict(x))
                val_scores = scores(valid.Labels.to_numpy(), prob.argmax(1))
                row = dict(fold=fold + 1, fraction=fraction, classifier=family, variant=variant,
                           train_rows=len(subset), train_groups=int(subset.group.nunique()),
                           validation_rows=len(valid), seconds=time.monotonic() - started,
                           **{'train_' + k: v for k, v in train_scores.items()}, **val_scores)
                records.append(row)
                if fraction == 1:
                    predictions[family + '__' + variant] = prob
                print(f"fold={fold+1} {family} fraction={fraction} {variant} F1={val_scores['macro_f1']:.4f} seconds={row['seconds']:.1f}", flush=True)
    return records, valid.row_id.to_numpy(), predictions, membership


def run(source, incumbent, output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a fresh destination')
    previous = json.loads((incumbent / 'protocol.json').read_text())
    assert file_sha256(source / 'dataset_with_splits.csv') == previous['source_data_sha256']
    assert file_sha256(source / 'emb_sapbert_concept.npy') == previous['embedding_sha256']
    families = json.loads((incumbent / 'family_selection.json').read_text())
    configs = {r['config']['classifier']: r['config'] for r in families}
    selection = json.loads((incumbent / 'selection.json').read_text())
    protocol = dict(source_data_sha256=previous['source_data_sha256'], embedding_sha256=previous['embedding_sha256'],
                    candidates=configs, fractions=FRACTIONS, folds=5, expected_fits=90, seed=20261001,
                    detail_version=VERSION, detail_feature_names=FEATURE_NAMES, detail_patterns=PATTERNS,
                    incumbent=selection, labels_changed=False, encoder_finetuned=False,
                    selection='Require development mean macro F1 greater than incumbent and emergency recall >= original baseline minus 0.01; rank by macro F1 then accuracy. No test-based selection.',
                    interpretation='Learning curves condition on previously selected settings and fixed grouped folds; no numeric extrapolation to 20,000 rows.')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print('AUDIT', json.dumps(audit_text_and_errors(source, incumbent, output)), flush=True)
    df = pd.read_csv(source / 'dataset_with_splits.csv')
    dev = df[df.partition.eq('development')]
    positions = {int(row): i for i, row in enumerate(dev.row_id)}
    oof = {f + '__' + variant: np.full((len(dev), 4), np.nan)
           for f in configs for variant in ['baseline', 'concept_details', 'complaint_details']}
    records, memberships = [], []
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(evaluate_fold, str(source.resolve()), fold, configs) for fold in range(5)]
        for job in as_completed(jobs):
            rows, ids, predictions, members = job.result()
            records.extend(rows)
            memberships.extend(members)
            for name, prob in predictions.items():
                oof[name][[positions[int(i)] for i in ids]] = prob
            pd.DataFrame(records).to_csv(output / 'cross_validation.csv', index=False)
            (output / 'memberships.json').write_text(json.dumps(memberships))
    np.savez_compressed(output / 'development_oof.npz', row_id=dev.row_id.to_numpy(), reference=dev.Labels.to_numpy(), **oof)
    assert len(records) == 90
    assert file_sha256(source / 'dataset_with_splits.csv') == protocol['source_data_sha256']
    assert file_sha256(source / 'emb_sapbert_concept.npy') == protocol['embedding_sha256']
    print('COMPLETE 90 development fits', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.incumbent, args.output)
