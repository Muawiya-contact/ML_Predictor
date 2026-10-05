"""Report each compared classifier family's best fused development configuration.

This descriptive comparison does not override the constrained deployment
selection. Freeze all family choices before computing retrospective scores.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
import research_engine as engine
from cache_integrity import file_sha256
from finalize_detail_comparison import fit_candidate
from four_level_study import scores
from sklearn.metrics import classification_report, confusion_matrix


def run(source, original, reuse=None):
    protocol = json.loads((source / 'protocol.json').read_text())
    assert file_sha256(original / 'dataset_with_splits.csv') == protocol['source_data_sha256']
    assert file_sha256(original / 'emb_sapbert_concept.npy') == protocol['embedding_sha256']
    summary = pd.read_csv(source / 'cv_summary.csv')
    choices = []
    for family in sorted({c['classifier'] for c in protocol['candidates'].values()}):
        names = [name for name, cfg in protocol['candidates'].items()
                 if cfg['classifier'] == family and cfg['features']['view'] == 'fused']
        best = summary[summary.candidate.isin(names)].sort_values(
            ['macro_f1', 'accuracy', 'candidate'], ascending=[False, False, True]).iloc[0]
        choices.append(dict(candidate=best.candidate, config=protocol['candidates'][best.candidate],
                            cv_macro_f1=float(best.macro_f1),
                            note='Best fused CV F1 within this family; descriptive comparison, not deployment selection.'))
    (source / 'family_selection.json').write_text(json.dumps(choices, indent=2))
    df = pd.read_csv(original / 'dataset_with_splits.csv')
    dev, test = df[df.partition.eq('development')], df[df.partition.eq('test')]
    assert not set(dev.group) & set(test.group)
    engine.HERE = original.resolve()
    results = []
    reusable = {}
    if reuse is not None:
        check = json.loads((reuse / 'verification.json').read_text())
        old_protocol = json.loads((reuse / 'protocol.json').read_text())
        if check['status'] != 'passed' or any(old_protocol[k] != protocol[k] for k in ('source_data_sha256','embedding_sha256')):
            raise ValueError('Reusable comparison inputs are not verified and identical')
        reusable = {r['candidate']: r for r in json.loads((reuse / 'family_results.json').read_text())}
    for choice in choices:
        cfg = choice['config']
        prior = reusable.get(choice['candidate'])
        if prior is not None and prior['config'] == cfg:
            results.append(dict(prior, **choice))
            filename = choice['candidate'] + '_family_predictions.csv'
            shutil.copy2(reuse / filename, source / filename)
            print('REUSED VERIFIED', choice['candidate'], flush=True)
            continue
        model, features, _, probabilities = fit_candidate(cfg, dev, test)
        pred = probabilities.argmax(1)
        result = dict(**choice, metrics=scores(test.Labels.to_numpy(), pred),
                      report=classification_report(test.Labels, pred, labels=[0,1,2,3], output_dict=True, zero_division=0),
                      confusion=confusion_matrix(test.Labels, pred, labels=[0,1,2,3]).tolist())
        results.append(result)
        pd.DataFrame(dict(row_id=test.row_id, reference=test.Labels, predicted=pred)).to_csv(
            source / (choice['candidate'] + '_family_predictions.csv'), index=False)
        print(choice['candidate'], result['metrics'], flush=True)
    (source / 'family_results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--reuse', type=Path)
    args = parser.parse_args()
    run(args.source, args.original, args.reuse)
