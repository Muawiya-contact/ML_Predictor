"""Validate the four-level workbook and recover only exactly matched concepts.

The previous CSV is an optional source of missing *input text*, never labels.
Every patient input and every already-populated concept must agree row-for-row
before recovery is allowed. Source files are never modified.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from cache_integrity import file_sha256

LABELS = {0: 'Emergency', 1: 'Urgent', 2: 'Standard', 3: 'Non-urgent'}
INPUTS = ['Age', 'Gender', 'Mode_of_Arrival', 'chief_complaint',
          'Heart_Rate', 'Systolic_BP', 'Diastolic_BP', 'ECG_Status',
          'Temperature', 'SpO2', 'AVPU']
PROVENANCE = ('Provider described label assignment as "by using all"; '
              'the exact process and independent per-record review are not documented.')


def prepare(source, output, reference=None):
    frame = pd.read_excel(source, sheet_name='Sheet1')
    required = INPUTS + ['Clinical_Concept', 'Triage_Level', 'Triage_Label']
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    target = pd.to_numeric(frame.Triage_Level, errors='coerce')
    if target.isna().any() or set(target.unique()) != set(LABELS):
        raise ValueError('Expected all four labels, numbered 0 through 3.')
    names = frame.Triage_Label.astype('string').str.strip().str.casefold()
    if not names.eq(target.map(LABELS).str.casefold()).all():
        raise ValueError('Triage_Level and Triage_Label disagree.')
    absent = frame.Clinical_Concept.isna() | frame.Clinical_Concept.astype(str).str.strip().eq('')
    count = int(absent.sum())
    metadata = {'file': source.name, 'sha256': file_sha256(source),
                'labels': LABELS, 'rows': len(frame), 'recovered_concepts': count,
                'label_provenance': PROVENANCE}
    if count:
        if reference is None:
            raise ValueError(f'{count} missing concepts; supply an exact-matching reference CSV.')
        old = pd.read_csv(reference)
        if len(old) != len(frame):
            raise ValueError('Reference row count differs; recovery refused.')
        for column in INPUTS:
            if not frame[column].astype(str).str.strip().eq(old[column].astype(str).str.strip()).all():
                raise ValueError(f'Reference inputs differ in {column}; recovery refused.')
        existing = ~absent
        if not frame.loc[existing, 'Clinical_Concept'].astype(str).str.strip().eq(
                old.loc[existing, 'Clinical_Concept'].astype(str).str.strip()).all():
            raise ValueError('Existing concepts differ from the reference; recovery refused.')
        frame.loc[absent, 'Clinical_Concept'] = old.loc[absent, 'Clinical_Concept']
        if frame.Clinical_Concept.isna().any() or frame.Clinical_Concept.astype(str).str.strip().eq('').any():
            raise ValueError('Reference does not resolve all missing concepts.')
        metadata.update(recovery_reference=reference.name,
                        recovery_reference_sha256=file_sha256(reference),
                        recovery_rule='All 11 complaint/patient inputs match in the same source row; existing concepts also match. Only missing Clinical_Concept values copied. New labels untouched.')
    # Target-name and processing metadata cannot enter the model design matrix.
    frame = frame[INPUTS + ['Clinical_Concept', 'Triage_Level']]
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Choose an empty output directory.')
    frame.to_csv(output / 'prepared_source.csv', index=False)
    metadata['prepared_sha256'] = file_sha256(output / 'prepared_source.csv')
    (output / 'source_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concept-reference', type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output, args.concept_reference)
