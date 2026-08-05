# prediction.py
# Load the trained triage model and predict triage level for new patients.
# Applies the shared pipeline from triage_pipeline.py:
#   normalization -> fuzzy matching -> diacritization -> BoW -> attention -> model
#
# Run:  python prediction.py
# (For predicting MANY patients from a file, use predict_batch.py instead.)

from triage_pipeline import load_artifacts, predict_one, TRIAGE_LABELS

print("Loading model...")
art = load_artifacts("triage_model")
print("[ok] Model and encoders loaded successfully.")


def predict_triage(complaint, age, heart_rate, systolic_bp, diastolic_bp,
                   temperature, spo2, gender, mode_of_arrival, avpu, ecg_status):
    """Thin wrapper around the shared single-patient predictor."""
    return predict_one(
        art, complaint, age, heart_rate, systolic_bp, diastolic_bp,
        temperature, spo2, gender, mode_of_arrival, avpu, ecg_status,
    )


if __name__ == "__main__":

    # --- Example patients (spanning several complaint categories) ---
    patients = [
        {
            "complaint":       "seena mein shadeed dard aur pasina aa raha hai",
            "age": 65, "heart_rate": 118, "systolic_bp": 160, "diastolic_bp": 95,
            "temperature": 37.2, "spo2": 94,
            "gender": "Male", "mode_of_arrival": "Ambulance",
            "avpu": "A", "ecg_status": "ST elevation",
        },
        {
            "complaint":       "saans phool rahi hai lekin baat kar sakti hai",
            "age": 45, "heart_rate": 112, "systolic_bp": 145, "diastolic_bp": 88,
            "temperature": 36.8, "spo2": 91,
            "gender": "Female", "mode_of_arrival": "Wheelchair",
            "avpu": "V", "ecg_status": "Abnormal",
        },
        {
            "complaint":       "bukhar aur body pain do din se",
            "age": 28, "heart_rate": 102, "systolic_bp": 128, "diastolic_bp": 78,
            "temperature": 38.9, "spo2": 97,
            "gender": "Male", "mode_of_arrival": "Walk-in",
            "avpu": "A", "ecg_status": "Normal",
        },
        {
            "complaint":       "mirgi ka daura para aur behosh ho gaya",
            "age": 33, "heart_rate": 96, "systolic_bp": 130, "diastolic_bp": 82,
            "temperature": 37.0, "spo2": 95,
            "gender": "Male", "mode_of_arrival": "Ambulance",
            "avpu": "P", "ecg_status": "Normal",
        },
        {
            "complaint":       "accident mein haddi toot gayi aur khoon beh raha hai",
            "age": 24, "heart_rate": 110, "systolic_bp": 105, "diastolic_bp": 70,
            "temperature": 36.6, "spo2": 96,
            "gender": "Male", "mode_of_arrival": "Ambulance",
            "avpu": "A", "ecg_status": "Normal",
        },
    ]

    print("\n" + "=" * 70)
    print("TRIAGE PREDICTIONS")
    print("=" * 70)

    for i, p in enumerate(patients, 1):
        level, confidence, proba = predict_triage(
            complaint=p["complaint"], age=p["age"], heart_rate=p["heart_rate"],
            systolic_bp=p["systolic_bp"], diastolic_bp=p["diastolic_bp"],
            temperature=p["temperature"], spo2=p["spo2"], gender=p["gender"],
            mode_of_arrival=p["mode_of_arrival"], avpu=p["avpu"],
            ecg_status=p["ecg_status"],
        )

        print(f"\nPatient {i}: \"{p['complaint']}\"")
        print(f"  -> Predicted Triage : {TRIAGE_LABELS[level]}")
        print(f"  -> Confidence       : {confidence * 100:.1f}%")
        print(f"  -> All probabilities: "
              f"L0={proba[0] * 100:.1f}%  L1={proba[1] * 100:.1f}%  "
              f"L2={proba[2] * 100:.1f}%  L3={proba[3] * 100:.1f}%")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
