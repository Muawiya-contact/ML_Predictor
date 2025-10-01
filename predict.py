import joblib
import pandas as pd

# Load model
model = joblib.load("patient_triage_model.pkl")

def predict_patient(patient_data):
    """
    patient_data: dict with patient features
    Example:
    {"SBP":120,"DBP":80,"HR":90,"RR":16,"BT":37.2,"SpO2":98,"Age":45,"Gender":1,
     "GCS":15,"Na":138,"K":4.1,"Cl":100,"Urea":40,"Ceratinine":1.1,
     "Alcoholic":0,"Smoke":0,"FHCD":0,"TriageScore":3}
    """


    df = pd.DataFrame([patient_data])
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    prediction = model.predict(df)[0]
    return prediction


if __name__ == "__main__":
    sample_patient = {
        "SBP": 120, "DBP": 80, "HR": 90, "RR": 16, "BT": 37.2, "SpO2": 98, "Age": 45,
        "Gender": 1, "GCS": 15, "Na": 138, "K": 4.1, "Cl": 100, "Urea": 40,
        "Ceratinine": 1.1, "Alcoholic": 0, "Smoke": 0, "FHCD": 0, "TriageScore": 3
    }
    print("Prediction:", predict_patient(sample_patient))


