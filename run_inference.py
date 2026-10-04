"""Local four-level triage using the same SapBERT bundle as the GUI.

Examples:
    python run_inference.py --check
    python run_inference.py 'seena mein dard hai' --age 65 --heart-rate 118
    python run_inference.py --interactive

Missing patient measurements are imputed by the fitted training pipeline and
reported in Notes. Translation and the anatomical gate run before scoring.
"""
import argparse
import contextlib
import json
import sys

import pandas as pd
from triage_pipeline import (get_text_encoder, has_text_signal, load_artifacts,
                             resolve_model_dir)
from predict_batch import predict_translated_dataframe
from src.offline_pipeline import ollama_available, ollama_models, select_translation_model


def check_environment(model_dir, model=None):
    path, _ = resolve_model_dir(model_dir)
    artifacts = load_artifacts(path)
    get_text_encoder(artifacts)
    print('Classifier:', artifacts['manifest']['method'])
    print('Classes:', artifacts['model'].classes_.tolist())
    print('Encoder: cached locally and ready')
    if not ollama_available():
        print('Ollama is not reachable; start ollama serve.')
        return False
    installed = ollama_models()
    selected = model or select_translation_model(installed)
    if model and model not in installed and model + ':latest' not in installed:
        print('Requested translator is not installed:', model)
        return False
    print('Local translator:', selected or 'none available')
    return selected is not None


def infer(artifacts, text, patient=None, model=None):
    if not has_text_signal(text):
        return {'Complaint_Text': text, 'Confidence': '50%',
                'Predicted_Triage_Level': None, 'Predicted_Label': None,
                'Notes': 'No usable complaint entered. 50% is a display placeholder, not a model prediction. Enter symptoms to obtain a triage level.'}
    frame = pd.DataFrame([dict(patient or {}, Complaint_Text=text)])
    result = predict_translated_dataframe(artifacts, frame, model=model)
    # pandas JSON handles NaN/null consistently for failed translations.
    return json.loads(result.to_json(orient='records'))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('complaint', nargs='*')
    parser.add_argument('--model-dir', help='Explicit compatible bundle; defaults to the GUI bundle')
    parser.add_argument('--model', help='Installed local Ollama model tag')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--json', action='store_true')
    fields = {'age': 'Age', 'heart-rate': 'Heart_Rate', 'systolic-bp': 'Systolic_BP',
              'diastolic-bp': 'Diastolic_BP', 'temperature': 'Temperature', 'spo2': 'SpO2',
              'gender': 'Gender', 'arrival': 'Mode_of_Arrival', 'avpu': 'AVPU', 'ecg': 'ECG_Status'}
    for option in fields:
        parser.add_argument('--' + option)
    args = parser.parse_args()
    if args.check:
        return 0 if check_environment(args.model_dir, args.model) else 1
    if not args.complaint and not args.interactive:
        parser.error('Provide a complaint, --interactive, or --check.')
    path, _ = resolve_model_dir(args.model_dir)
    artifacts = load_artifacts(path)
    patient = {name: getattr(args, flag.replace('-', '_')) for flag, name in fields.items()}

    def display(text):
        with contextlib.redirect_stdout(sys.stderr):
            result = infer(artifacts, text, patient, args.model)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for key in ['Complaint_Text', 'Translation_English', 'Gate_Status',
                        'Predicted_Triage_Level', 'Predicted_Label', 'Confidence', 'Notes']:
                if key in result:
                    print(f'{key}: {result[key]}')
        return result.get('Predicted_Triage_Level') is not None or not has_text_signal(text)

    if not args.interactive:
        return 0 if display(' '.join(args.complaint)) else 1
    print('Four-level local triage. Enter exit to quit.')
    while True:
        try:
            text = input('Complaint: ')
        except (EOFError, KeyboardInterrupt):
            break
        if text.casefold() in ('exit', 'quit'):
            break
        display(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
