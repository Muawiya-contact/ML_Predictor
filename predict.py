import joblib
import pandas as pd
from flask import request
import json
import numpy as np

FEATURES = [
    "SBP", "DBP", "HR", "RR", "BT", "SpO2", "Age", "Gender", "GCS",
    "Na", "K", "Cl", "Urea", "Ceratinine",
    "Alcoholic", "Smoke", "FHCD", "TriageScore"
]

model_path = "patient_triage_model.pkl"

def prepare_features(patient_data: dict):
    # Ensure all required features are present
    return {feat: float(patient_data.get(feat, 0)) for feat in FEATURES}

def predict_patient(patient_data, model_path=model_path, features=FEATURES):
    model, preprocessor = joblib.load(model_path)
    X = np.array([[patient_data[feat] for feat in features]])
    X_proc = preprocessor.transform(X)
    pred = model.predict(X_proc)[0]
    return pred

def save_history():
    with open("history.json", "w") as f:
        json.dump(history, f)

def load_history():
    global history
    try:
        with open("history.json") as f:
            history = json.load(f)
    except:
        history = []



def predict():
    try:
        if request.is_json:
            data = request.get_json(force=True)
        else:
            data = request.form.to_dict()

        # Normalize keys to match FEATURES
        data = {k.capitalize(): v for k, v in data.items()}  # or use exact mapping

        patient_features = prepare_features(data)
        prediction = predict_patient(patient_features)
        explanation = get_gpt_explanation(prediction, patient_features)
        # ... rest of the code ...
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    load_history()
    sample_patient = {
        "SBP": 120, "DBP": 80, "HR": 90, "RR": 16, "BT": 37.2, "SpO2": 98, "Age": 45,
        "Gender": 1, "GCS": 15, "Na": 138, "K": 4.1, "Cl": 100, "Urea": 40,
        "Ceratinine": 1.1, "Alcoholic": 0, "Smoke": 0, "FHCD": 0, "TriageScore": 3
    }
    print("Prediction:", predict_patient(sample_patient))
