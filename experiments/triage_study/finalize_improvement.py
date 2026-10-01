"""Freeze joint development selection across the main and interaction searches."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import research_engine as engine
from improve_four_level import rank
from four_level_study import scores
from cache_integrity import file_sha256

def finalize(main, extra, original, output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a fresh final destination')
    protocol = json.loads((main / 'protocol.json').read_text())
    additional = json.loads((extra / 'protocol.json').read_text())
    if protocol['source_data_sha256'] != additional['source_data_sha256'] or file_sha256(original / 'dataset_with_splits.csv') != protocol['source_data_sha256']:
        raise ValueError('Source identity mismatch')
    if file_sha256(original / 'emb_sapbert_concept.npy') != protocol['embedding_sha256']:
        raise ValueError('Embedding identity mismatch')
    if set(protocol['candidates']) & set(additional['candidates']):
        raise ValueError('Candidate names must be unique across rounds')
    if additional.get('embedding_sha256', protocol['embedding_sha256']) != protocol['embedding_sha256']:
        raise ValueError('Supplement embedding mismatch')
    protocol['candidates'].update(additional['candidates'])
    protocol['label_assignment_rules'] = 'Provider has none available; all supplied labels remain unchanged.'
    protocol['supplement'] = additional.get('supplement', 'Six quadratic patient-feature LR configurations predeclared separately; joint choice uses development scores only.')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    cv = pd.concat([pd.read_csv(main / 'cross_validation.csv'), pd.read_csv(extra / 'cross_validation.csv')], ignore_index=True)
    if len(cv) != 5 * len(protocol['candidates']) or cv.duplicated(['candidate', 'fold']).any():
        raise ValueError('Incomplete/duplicate CV fits')
    cv.to_csv(output / 'cross_validation.csv', index=False)
    summary = cv.groupby('candidate').agg(macro_f1=('macro_f1', 'mean'), f1_std=('macro_f1', 'std'), accuracy=('accuracy', 'mean'), precision=('precision_macro', 'mean'), recall=('recall_macro', 'mean'), emergency_recall=('emergency_recall', 'mean'), under_triage=('under_triage_rate', 'mean')).reset_index()
    baseline = summary.set_index('candidate').loc[protocol['baseline']]
    chosen = rank(summary, baseline).iloc[0]
    selection = dict(candidate=chosen.candidate, config=protocol['candidates'][chosen.candidate], baseline=protocol['baseline'], rule=protocol['selection'], cv_gain=float(chosen.macro_f1 - baseline.macro_f1))
    summary.sort_values('macro_f1', ascending=False).to_csv(output / 'cv_summary.csv', index=False)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2))
    print('FROZEN JOINT SELECTION', selection, flush=True)
    previous = json.loads((main / 'selection.json').read_text())
    oof = np.load(main / 'development_oof.npz')
    other = np.load(extra / 'oof.npz')
    np.testing.assert_array_equal(oof['row_id'], other['row_id'])
    if chosen.candidate == previous['candidate']:
        p = oof['selected']
    elif chosen.candidate in other.files:
        p = other[chosen.candidate]
    else:
        raise ValueError('Selected candidate OOF predictions are unavailable')
    np.savez_compressed(output / 'development_oof.npz', row_id=oof['row_id'], reference=oof['reference'], selected=p, baseline=oof['baseline'])
    df = pd.read_csv(original / 'dataset_with_splits.csv')
    dev = df[df.partition.eq('development')].copy()
    test = df[df.partition.eq('test')]
    np.testing.assert_array_equal(dev.row_id, oof['row_id'])
    dev['oof_predicted'] = p.argmax(1)
    dev['oof_confidence'] = p.max(1)
    dev[dev.Labels.ne(dev.oof_predicted)].to_csv(output / 'development_errors_for_review.csv', index=False)
    shutil.copy2(main / 'input_audit.json', output / 'input_audit.json')
    engine.HERE = original.resolve()
    config = selection['config']
    transform = engine.Features(**config['features']).fit(dev)
    model = engine.fit_model(config, transform.transform(dev), dev.Labels.to_numpy())
    pred = model.predict_proba(transform.transform(test)).argmax(1)
    result = dict(candidate=chosen.candidate, feature_count=int(model.n_features_in_), pca_retained_variance=float(transform.pca_.explained_variance_ratio_.sum()) if transform.pca_ is not None else None, metrics=scores(test.Labels.to_numpy(), pred), report=classification_report(test.Labels, pred, labels=[0, 1, 2, 3], output_dict=True, zero_division=0), confusion=confusion_matrix(test.Labels, pred, labels=[0, 1, 2, 3]).tolist())
    old_results = json.loads((main / 'retrospective_results.json').read_text())
    previous_result = next(r for r in old_results if r['candidate'] == previous['candidate'])
    (output / 'previous_round_result.json').write_text(json.dumps(previous_result, indent=2))
    base = next((r for r in old_results if r['candidate'] == protocol['baseline']))
    (output / 'retrospective_results.json').write_text(json.dumps([base, result] if chosen.candidate != protocol['baseline'] else [base], indent=2))
    shutil.copy2(main / (protocol['baseline'] + '_retrospective_predictions.csv'), output / (protocol['baseline'] + '_retrospective_predictions.csv'))
    pd.DataFrame(dict(row_id=test.row_id, reference=test.Labels, predicted=pred)).to_csv(output / (chosen.candidate + '_retrospective_predictions.csv'), index=False)
    dest = output / 'serving'
    dest.mkdir()
    for name, value in [('model', model), ('structured', transform.structured_), ('pca', transform.pca_)]:
        joblib.dump(value, dest / (name + '.pkl'))
    print('COMPLETE joint comparison', flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--main', type=Path, required=True)
    p.add_argument('--extra', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    finalize(a.main, a.extra, a.original, a.output)
