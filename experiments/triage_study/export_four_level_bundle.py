"""Export the CV-selected four-level model without fitting or changing scores."""
import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from cache_integrity import file_sha256
from prepare_four_level_source import PROVENANCE

NAMES = {'logreg': 'Logistic Regression', 'hgb': 'Hist Gradient Boosting', 'rf': 'Random Forest'}


def main(source, output):
    selected = json.loads((source / 'selected_results.json').read_text())
    plan = json.loads((source / 'comparison_plan.json').read_text())
    selection = json.loads((source / 'selection.json').read_text())
    if selection['config'] != selected['config'] or plan['labels'] != [0, 1, 2, 3]:
        raise ValueError('Selected model and frozen four-level plan differ.')
    if file_sha256(source / 'dataset_with_splits.csv') != plan['source_data_sha256']:
        raise ValueError('Dataset changed since training.')
    if file_sha256(source / 'emb_sapbert_concept.npy') != plan['embedding_sha256']:
        raise ValueError('Embeddings changed since training.')
    frame = pd.read_csv(source / 'dataset_with_splits.csv')
    test = frame[frame.partition.eq('test')]
    import research_engine as engine
    engine.HERE = source.resolve()
    transform = engine.Features(**selected['config']['features'])
    transform.structured_ = joblib.load(source / 'serving/structured.pkl')
    transform.pca_ = joblib.load(source / 'serving/pca.pkl')
    model = joblib.load(source / 'serving/model.pkl')
    if model.classes_.tolist() != [0, 1, 2, 3]:
        raise ValueError('Wrong classifier classes.')
    actual = model.predict(transform.transform(test))
    expected = pd.read_csv(source / 'selected_fused_64_predictions.csv')
    np.testing.assert_array_equal(test.row_id, expected.row_id)
    np.testing.assert_array_equal(actual, expected.predicted)
    np.testing.assert_array_equal(confusion_matrix(test.Labels, actual, labels=[0, 1, 2, 3]), selected['confusion'])
    metadata = json.loads((source / 'source_metadata.json').read_text())
    encoder = plan['encoder']
    root = Path(__file__).resolve().parents[2]
    snapshot = Path(encoder['model_path'])
    try:
        snapshot = snapshot.relative_to(root)
    except ValueError:
        pass
    manifest = {
        'backend': 'sapbert_pca',
        'method': 'SapBERT + PCA-64 + ' + NAMES[selected['config']['classifier']],
        'labels': [0, 1, 2, 3], 'label_names': plan['label_names'],
        'text_representation': 'embeddings_raw',
        'embedding_model': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
        'embedding_dim': 768, 'projected_embedding_dim': 64,
        'feature_blocks': [{'name': 'structured', 'dim': int(model.n_features_in_ - 64)},
                           {'name': 'embedding', 'dim': 64, 'rescaled': False}],
        'encoder_settings': {'revision': encoder['revision'], 'pooling': 'cls',
                             'max_token_length': 64, 'normalized': True, 'local_path': str(snapshot)},
        'text_pipeline': 'English -> SapBERT CLS (768) -> fitted PCA (64)',
        'text_column': 'Clinical_Concept', 'experiment': True,
        'dataset': {'file': metadata['file'], 'sha256': metadata['sha256'], 'rows': len(frame),
                    'training_rows': plan['development_rows'], 'test_rows': plan['test_rows'],
                    'n_classes': 4, 'recovered_concepts': metadata['recovered_concepts'],
                    'provenance': {'label_method': PROVENANCE}},
        'scope': {'clinical_scope': 'CARDIAC COMPLAINTS; LEVELS 0-3'},
        'evaluation_note': 'Saved metrics evaluate supplied clinical concepts, not live Ollama translations. These are research results, not clinical validation.',
        'source_config': selected['config'], 'selection': selection,
        'source_data_sha256': plan['source_data_sha256'],
        'artifact_sha256': {name: file_sha256(source / 'serving' / name)
                            for name in ['model.pkl', 'structured.pkl', 'pca.pkl']},
    }
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use an empty export directory; promotion is a separate step.')
    for name in manifest['artifact_sha256']:
        shutil.copy2(source / 'serving' / name, output / name)
    (output / 'model_manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'learned_stopwords.json').write_text(json.dumps({'stopwords': [], 'note': 'This encoder uses unfiltered English text.'}))
    metrics = {'accuracy': 100 * selected['metrics']['accuracy'], 'labels': [0, 1, 2, 3],
               'confusion_matrix': selected['confusion'], 'metrics': selected['metrics'],
               'classification_report': selected['report'], 'evaluation_note': manifest['evaluation_note']}
    (output / 'triage_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f'Exported and verified all {len(test)} held-out predictions: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
