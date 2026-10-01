"""Live SapBERT inference using exported, fitted research preprocessing.

No row-index lookup or refitting occurs at prediction time. The encoder uses
exactly the study's CLS pooling, 64-token truncation and L2 normalization.
"""
import os
import hashlib
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUM = ['Age', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP', 'Temperature', 'SpO2', 'avpu_ord']
CAT = ['Gender', 'Mode_of_Arrival', 'ECG_Status']
AVPU = {'a': 0, 'alert': 0, 'v': 1, 'voice': 1, 'verbal': 1,
        'p': 2, 'pain': 2, 'u': 3, 'unresponsive': 3}


class SapBERTEncoder:
    """Offline-only adapter exposing the GUI's existing encode interface."""
    def __init__(self, manifest):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(2)
        settings = manifest['encoder_settings']
        location = os.environ.get('SAPBERT_MODEL_PATH', settings['local_path'])
        path = Path(location)
        if not path.is_absolute():
            path = ROOT / path
        # A cloned repository may use the standard Hugging Face cache instead
        # of the original local snapshot directory. Both paths stay offline.
        location = str(path) if path.exists() or os.environ.get('SAPBERT_MODEL_PATH') else manifest['embedding_model']
        options = {'local_files_only': True, 'revision': settings['revision']}
        self.tokenizer = AutoTokenizer.from_pretrained(location, **options)
        self.model = AutoModel.from_pretrained(location, **options).eval()
        self.max_length = settings['max_token_length']

    def get_sentence_embedding_dimension(self):
        return 768

    def encode(self, texts, batch_size=16, **kwargs):
        import torch
        if isinstance(texts, str):
            return self.encode([texts], batch_size=batch_size, **kwargs)[0]
        if not texts:
            return np.empty((0, 768), dtype=np.float32)
        pieces = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                tokens = self.tokenizer(list(texts[start:start + batch_size]),
                                        padding=True, truncation=True,
                                        max_length=self.max_length, return_tensors='pt')
                values = self.model(**tokens).last_hidden_state[:, 0, :].cpu().numpy()
                values /= np.linalg.norm(values, axis=1, keepdims=True).clip(1e-9)
                pieces.append(values)
        return np.vstack(pieces)


def prepare(frame):
    """Mirror the study's numeric conversion and ordinal AVPU mapping."""
    frame = frame.copy()
    frame['avpu_ord'] = frame.AVPU.astype(str).str.lower().map(AVPU)
    for name in NUM:
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
    for name in CAT:
        frame[name] = frame[name].where(frame[name].notna(), np.nan).astype(object)
    return frame


def load_bundle(model_dir, manifest):
    import sklearn
    if manifest.get('sklearn_version') != sklearn.__version__:
        raise ValueError('SapBERT bundle requires scikit-learn ' + str(manifest.get('sklearn_version')))
    path = Path(model_dir)
    files = ['model.pkl', 'structured.pkl', 'pca.pkl']
    details = manifest.get('text_details')
    if details:
        from src.complaint_details import VERSION, FEATURE_NAMES
        if (details.get('version') != VERSION or details.get('feature_names') != FEATURE_NAMES
                or details.get('source') not in ('concept_details', 'complaint_details')):
            raise ValueError('Text-detail definition does not match the evaluated manifest')
        files.append('detail_scaler.pkl')
    for name in files:
        expected = manifest.get('artifact_sha256', {}).get(name)
        if expected is None or hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'SapBERT artifact does not match its evaluated manifest: {name}')
    art = {'model': joblib.load(path / 'model.pkl'),
           'structured': joblib.load(path / 'structured.pkl'),
           'pca': joblib.load(path / 'pca.pkl'),
           'manifest': manifest, 'model_dir': str(path),
           'text_representation': 'embeddings_raw', 'blocks': ('embedding',),
           'encoder': None, 'stopwords': set()}
    if details:
        art['detail_scaler'] = joblib.load(path / 'detail_scaler.pkl')
        if art['detail_scaler'].n_features_in_ != len(FEATURE_NAMES):
            raise ValueError('Text-detail scaler dimensions do not match the manifest')
    if art['model'].n_features_in_ != sum(b['dim'] for b in manifest['feature_blocks']):
        raise ValueError('SapBERT classifier dimensions do not match its manifest')
    if list(art['model'].classes_) != [0, 1, 2, 3] or manifest.get('labels') != [0, 1, 2, 3]:
        raise ValueError('SapBERT serving bundle must have classes 0, 1, 2 and 3')
    projected = manifest.get('projected_embedding_dim')
    if (projected not in (64, 128, 256) or art['pca'].n_features_in_ != 768
            or art['pca'].n_components_ != projected):
        raise ValueError('SapBERT PCA dimensions do not match the evaluated manifest')
    cats = art['structured'].named_transformers_['cat'].named_steps['encoder'].categories_
    for key, values in zip(['le_gender', 'le_mode', 'le_ecg'], cats):
        art[key] = SimpleNamespace(classes_=values)
    art['le_avpu'] = SimpleNamespace(classes_=np.array(['A', 'V', 'P', 'U']))
    return art


def predict_frame(art, frame):
    """Return existing GUI output schema, preserving cap and quality notes."""
    from triage_pipeline import (build_text_features, has_text_signal,
                                 MAX_CONFIDENCE_WITHOUT_TEXT, NO_TEXT_SIGNAL_WARNING,
                                 TRIAGE_LABELS)
    frame = frame.copy()
    for name in NUM[:6] + CAT + ['AVPU', 'Complaint_Text']:
        if name not in frame:
            frame[name] = np.nan
    notes = [[] for _ in range(len(frame))]
    prepared = prepare(frame)
    for pos, (_, row) in enumerate(prepared.iterrows()):
        for name in NUM:
            if pd.isna(row[name]):
                notes[pos].append(f'{name} missing/unreadable -> training median')
        for name, categories in zip(CAT, art['structured'].named_transformers_['cat'].named_steps['encoder'].categories_):
            if pd.isna(row[name]):
                notes[pos].append(f'{name} missing -> training most frequent value')
            elif row[name] not in categories:
                notes[pos].append(f'{name} not recognised -> zero categorical indicators')
    texts = frame.Complaint_Text.fillna('').astype(str).tolist()
    if len(frame):
        structured = art['structured'].transform(prepared)
        blocks = [structured, build_text_features(art, texts)]
        details = art.get('manifest', {}).get('text_details')
        if details:
            from src.complaint_details import detail_matrix
            column = 'Raw_Complaint' if details['source'] == 'complaint_details' else 'Complaint_Text'
            if column not in frame or frame[column].isna().any():
                raise ValueError(f'{column} is required by this evaluated text-detail model; do not substitute translated text for the original complaint.')
            blocks.append(art['detail_scaler'].transform(detail_matrix(frame[column].astype(str).tolist())))
        X = np.hstack(blocks)
        probabilities = art['model'].predict_proba(X)
    else:
        probabilities = np.empty((0, 4))
    indices = probabilities.argmax(axis=1)
    levels = art['model'].classes_[indices].astype(int)
    confidences = []
    for pos, text in enumerate(texts):
        confidence = float(probabilities[pos, indices[pos]])
        if not has_text_signal(text):
            notes[pos].insert(0, NO_TEXT_SIGNAL_WARNING)
            confidence = min(confidence, MAX_CONFIDENCE_WITHOUT_TEXT)
        confidences.append(confidence)
    out = frame.copy()
    out['Predicted_Level_0to3'] = levels
    out['Predicted_Triage_Level'] = levels
    out['Predicted_Label'] = [TRIAGE_LABELS[int(v)].split('(')[0].strip() for v in levels]
    out['Confidence'] = [f'{v * 100:.1f}%' for v in confidences]
    for idx in range(4):
        out[f'P_L{idx}'] = [f'{v * 100:.1f}%' for v in probabilities[:, idx]]
    out['Notes'] = ['; '.join(n) for n in notes]
    return out, notes, probabilities, confidences


def embed_step(art, text, translate=True):
    """Use the active encoder for the GUI's similarity and cluster panels."""
    from triage_pipeline import get_text_encoder
    from src.offline_pipeline import (fuzzy_normalize_roman_urdu,
                                     translate_roman_urdu, verify_anatomical_integrity)
    normalized = fuzzy_normalize_roman_urdu(text, verbose=False) if translate else text
    english = translate_roman_urdu(normalized) if translate else text
    if not english:
        raise ValueError('Local translation failed; no embedding produced.')
    if translate:
        passed, failures = verify_anatomical_integrity(normalized, english)
        if not passed:
            raise ValueError('Anatomical check failed: ' + '; '.join(failures))
    vector = get_text_encoder(art).encode([english])[0]
    return {'raw': text, 'translated': english, 'normalized': english,
            'embedding': vector, 'encoder': art['manifest']['embedding_model'],
            'l2_norm': float(np.linalg.norm(vector)), 'translated_ok': bool(translate),
            'error': None}
