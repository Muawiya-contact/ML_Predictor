"""Check exported classifier parity through the shared live application adapter."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from triage_pipeline import load_artifacts, get_text_encoder, predict_dataframe, predict_one

def verify(source, original, bundle):
    selection = json.loads((source / 'selection.json').read_text())
    frame = pd.read_csv(original / 'dataset_with_splits.csv')
    test = frame[frame.partition.eq('test')].copy()
    reference = pd.read_csv(source / (selection['candidate'] + '_retrospective_predictions.csv'))
    raw = np.load(original / 'emb_sapbert_concept.npy', mmap_mode='r')
    art = load_artifacts(str(bundle))
    encoder = get_text_encoder(art)
    indices = frame.groupby('Labels').head(3).index.tolist() + [frame.Clinical_Concept.str.len().idxmax()]
    live = encoder.encode(frame.loc[indices].Clinical_Concept.tolist())
    np.testing.assert_allclose(live, raw[indices], atol=2e-06, rtol=2e-05)

    class Cached:

        def encode(self, texts, **kwargs):
            assert texts == test.Clinical_Concept.tolist()
            return raw[test.row_id]
    art['encoder'] = Cached()
    test['Complaint_Text'] = test.Clinical_Concept
    test['Raw_Complaint'] = test.chief_complaint
    out, _ = predict_dataframe(art, test)
    probability_file = source / 'selected_probabilities.npy'
    if probability_file.exists():
        from src.sapbert_serving import predict_frame
        _, _, probabilities, _ = predict_frame(art, test)
        np.testing.assert_allclose(probabilities, np.load(probability_file), atol=1e-12)
    np.testing.assert_array_equal(test.row_id, reference.row_id)
    np.testing.assert_array_equal(out.Predicted_Triage_Level, reference.predicted)
    assert all((f'P_L{i}' in out for i in range(4)))
    art['encoder'] = encoder
    row = test.iloc[0]
    args = [row.Clinical_Concept] + [row[c] for c in ['Age', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP', 'Temperature', 'SpO2', 'Gender', 'Mode_of_Arrival', 'AVPU', 'ECG_Status']]
    level, confidence, proba = predict_one(art, *args, raw_complaint=row.chief_complaint)
    assert level == reference.predicted.iloc[0] and len(proba) == 4
    np.testing.assert_allclose(proba.sum(), 1)
    result = dict(status='passed', retrospective_predictions=len(test), live_embedding_checks=len(indices), live_single_prediction='passed', live_translation_tested=False)
    (source / 'serving_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    a = p.parse_args()
    verify(a.source, a.original, a.bundle)
