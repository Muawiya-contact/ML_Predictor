"""Independently verify group membership, reproduction and development selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from four_level_study import scores
from cache_integrity import file_sha256


def verify(source, original, incumbent):
    protocol = json.loads((source / 'protocol.json').read_text())
    for filename, key in [('dataset_with_splits.csv','source_data_sha256'), ('emb_sapbert_concept.npy','embedding_sha256')]:
        assert file_sha256(original / filename) == protocol[key]
    df = pd.read_csv(original / 'dataset_with_splits.csv')
    dev = df[df.partition.eq('development')]
    lookup = df.set_index('row_id')
    memberships = json.loads((source / 'memberships.json').read_text())
    assert len(memberships) == 20
    for fold in range(1,6):
        before = set()
        members = sorted([m for m in memberships if m['fold']==fold], key=lambda m:m['fraction'])
        assert [m['fraction'] for m in members] == [.25,.5,.75,1.]
        expected_train = set(dev.loc[dev.cv5.ne(fold-1),'row_id'])
        expected_validation = set(dev.loc[dev.cv5.eq(fold-1),'row_id'])
        for member in members:
            ids = set(member['train_row_ids'])
            assert before <= ids <= expected_train
            assert set(member['validation_row_ids']) == expected_validation
            a, b = lookup.loc[list(ids)], lookup.loc[list(expected_validation)]
            assert not set(a.group) & set(b.group)
            # Entire groups, not individual members, must enter each subset.
            full_group_ids = set(dev.loc[dev.group.isin(a.group.unique()),'row_id'])
            assert full_group_ids == ids
            if member['fraction'] == 1:
                assert ids == expected_train
            before = ids
    cv = pd.read_csv(source / 'cross_validation.csv')
    assert len(cv)==90 and not cv.duplicated(['fold','fraction','classifier','variant']).any()
    expected = {(fold, fraction, family, variant) for fold in range(1,6)
                for fraction in [.25,.5,.75,1.] for family in ['logreg','hgb','rf']
                for variant in (['baseline'] if fraction != 1 else ['baseline','concept_details','complaint_details'])}
    assert set(cv[['fold','fraction','classifier','variant']].itertuples(index=False,name=None)) == expected
    oof = np.load(source / 'development_oof.npz')
    np.testing.assert_array_equal(oof['row_id'],dev.row_id)
    np.testing.assert_array_equal(oof['reference'],dev.Labels)
    old_cv = pd.read_csv(incumbent / 'cross_validation.csv')
    family_choices = json.loads((incumbent / 'family_selection.json').read_text())
    for choice in family_choices:
        family = choice['config']['classifier']
        old = old_cv[old_cv.candidate.eq(choice['candidate'])].sort_values('fold')
        current = cv[cv.classifier.eq(family)&cv.variant.eq('baseline')&cv.fraction.eq(1)].sort_values('fold')
        for metric in ['accuracy','precision_macro','recall_macro','macro_f1','emergency_recall']:
            np.testing.assert_allclose(old[metric], current[metric], atol=1e-12)
        for variant in ['baseline','concept_details','complaint_details']:
            probabilities = oof[family+'__'+variant]
            assert np.isfinite(probabilities).all() and (probabilities>=0).all()
            np.testing.assert_allclose(probabilities.sum(1),1,atol=1e-10)
            for fold in range(5):
                mask = dev.cv5.to_numpy()==fold
                actual = scores(dev.Labels.to_numpy()[mask], probabilities[mask].argmax(1))
                record = cv[cv.classifier.eq(family)&cv.variant.eq(variant)&cv.fraction.eq(1)&cv.fold.eq(fold+1)].iloc[0]
                for key,value in actual.items():
                    np.testing.assert_allclose(record[key], value, atol=1e-12)
    summary = cv.groupby(['classifier','variant','fraction']).agg(
        train_rows=('train_rows','mean'), train_f1=('train_macro_f1','mean'),
        train_accuracy=('train_accuracy','mean'), macro_f1=('macro_f1','mean'),
        f1_std=('macro_f1','std'), accuracy=('accuracy','mean'), precision=('precision_macro','mean'),
        recall=('recall_macro','mean'), emergency_recall=('emergency_recall','mean'),
        under_triage=('under_triage_rate','mean')).reset_index()
    summary.to_csv(source/'summary.csv',index=False)
    reference = summary[summary.classifier.eq('logreg')&summary.variant.eq('baseline')&summary.fraction.eq(1)].iloc[0]
    original_baseline = protocol['incumbent']['baseline']
    threshold = float(old_cv[old_cv.candidate.eq(original_baseline)].emergency_recall.mean()-.01)
    eligible = summary[summary.fraction.eq(1)&summary.emergency_recall.ge(threshold)&summary.macro_f1.ge(reference.macro_f1)]
    chosen = eligible.sort_values(['macro_f1','accuracy','classifier','variant'],ascending=[False,False,True,True]).iloc[0]
    selection = dict(classifier=chosen.classifier,variant=chosen.variant,
                     config=protocol['candidates'][chosen.classifier],
                     cv_macro_f1=float(chosen.macro_f1),incumbent_cv_macro_f1=float(reference.macro_f1),
                     cv_gain=float(chosen.macro_f1-reference.macro_f1),
                     minimum_emergency_recall=threshold,emergency_recall=float(chosen.emergency_recall),
                     uses_test_scores=False)
    (source/'selection.json').write_text(json.dumps(selection,indent=2))
    result = dict(status='passed',verified_fits=90,verified_learning_memberships=20,
                  full_size_baselines_reproduced=True,source_hashes_verified=True,
                  untouched_labels=True,test_rows_used=False,
                  interpretation='Development diagnostics conditional on earlier model selection; not an independent estimate.')
    (source/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(selection,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--incumbent',type=Path,required=True)
    args=parser.parse_args()
    verify(args.source,args.original,args.incumbent)
