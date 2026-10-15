"""Independently check every saved four-level score and its leakage boundary."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from cache_integrity import file_sha256
from prepare_data import normal


def verify(root):
    plan = json.loads((root / 'comparison_plan.json').read_text())
    frame = pd.read_csv(root / 'dataset_with_splits.csv')
    assert file_sha256(root / 'dataset_with_splits.csv') == plan['source_data_sha256']
    assert file_sha256(root / 'emb_sapbert_concept.npy') == plan['embedding_sha256']
    development = frame[frame.partition.eq('development')]
    test = frame[frame.partition.eq('test')]
    assert len(development) == plan['development_rows'] and len(test) == plan['test_rows']
    assert set(development.Labels) == set(test.Labels) == {0, 1, 2, 3}
    assert set(development.group).isdisjoint(test.group)
    for column in ['chief_complaint', 'Clinical_Concept']:
        a = set(development[column].map(normal)) - {''}
        b = set(test[column].map(normal)) - {''}
        assert a.isdisjoint(b), f'{column} crosses the holdout boundary'
    for fold in range(5):
        fitting = development[development.cv5.ne(fold)]
        validation = development[development.cv5.eq(fold)]
        assert set(fitting.group).isdisjoint(validation.group)
        assert set(fitting.Labels) == set(validation.Labels) == {0, 1, 2, 3}
    cv = pd.read_csv(root / 'cross_validation.csv')
    assert len(cv) == 50 and not cv.duplicated(['candidate', 'fold']).any()
    for candidate in plan['cv_candidates']:
        assert set(cv[cv.candidate.eq(candidate)].fold) == {1, 2, 3, 4, 5}
    ranking = cv.groupby('candidate').agg(
        macro_f1_mean=('macro_f1', 'mean'), emergency_recall_mean=('emergency_recall', 'mean'),
        under_triage_mean=('under_triage_rate', 'mean')).reset_index().sort_values(
            ['macro_f1_mean', 'emergency_recall_mean', 'under_triage_mean', 'candidate'], ascending=[False, False, True, True])
    selection = json.loads((root / 'selection.json').read_text())
    assert selection['candidate'] == ranking.iloc[0].candidate
    selected = json.loads((root / 'selected_results.json').read_text())
    assert selection['config'] == selected['config']
    rows = json.loads((root / 'results.json').read_text())
    assert {r['id'] for r in rows} == {f'{view}_{pc}_{clf}' for view in ['text', 'fused']
                                         for pc in [768, 64] for clf in ['logreg', 'hgb', 'rf']}
    for result in rows + [selected]:
        predicted = pd.read_csv(root / (result['id'] + '_predictions.csv'))
        np.testing.assert_array_equal(predicted.row_id, test.row_id)
        np.testing.assert_array_equal(predicted.reference, test.Labels)
        assert set(predicted.predicted) <= {0, 1, 2, 3}
        matrix = confusion_matrix(predicted.reference, predicted.predicted, labels=[0, 1, 2, 3])
        np.testing.assert_array_equal(matrix, result['confusion'])
        report = classification_report(predicted.reference, predicted.predicted, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)
        expected = {'accuracy': accuracy_score(predicted.reference, predicted.predicted),
                    'precision_macro': report['macro avg']['precision'],
                    'recall_macro': report['macro avg']['recall'],
                    'macro_f1': report['macro avg']['f1-score'], 'emergency_recall': report['0']['recall'],
                    'under_triage_rate': float((predicted.predicted > predicted.reference).mean()),
                    'over_triage_rate': float((predicted.predicted < predicted.reference).mean())}
        for metric, value in expected.items():
            assert np.isclose(value, result['metrics'][metric], atol=1e-12), (result['id'], metric)
    # Group bootstrap on the fixed selected model quantifies sampling uncertainty.
    # It never feeds back into model selection or training.
    prediction = pd.read_csv(root / 'selected_fused_64_predictions.csv').predicted.to_numpy()
    y = test.Labels.to_numpy()
    groups = test.group.to_numpy()
    positions = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    rng = np.random.default_rng(2026)
    scores = []
    for _ in range(1000):
        idx = np.concatenate([positions[i] for i in rng.integers(len(positions), size=len(positions))])
        scores.append([accuracy_score(y[idx], prediction[idx]),
                       f1_score(y[idx], prediction[idx], labels=[0, 1, 2, 3], average='macro', zero_division=0)])
    ci = {metric: np.percentile(np.array(scores)[:, i], [2.5, 97.5]).tolist()
          for i, metric in enumerate(['accuracy', 'macro_f1'])}
    result = {'verified_conditions': 13, 'verified_cv_fits': 50, 'shared_holdout_groups': 0,
              'test_rows': len(test), 'selected_candidate': selection['candidate'],
              'selected_group_bootstrap_95_ci': ci, 'bootstrap_replicates': 1000,
              'bootstrap_seed': 2026, 'status': 'passed'}
    (root / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    verify(parser.parse_args().results)
