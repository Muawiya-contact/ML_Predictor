# prediction_interactive.py
# Interactive terminal-based triage prediction (one patient at a time).
# Uses the shared pipeline from triage_pipeline.py.
# Run: python prediction_interactive.py

import os
import sys

# See prediction.py: without this, `python -m prediction_interactive` and
# IDE runners fail on "No module named 'triage_pipeline'".
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from triage_pipeline import (
    describe_model,
    load_artifacts,
    predict_one,
    normalize_roman_urdu,
    make_console_safe,
    resolve_model_dir,
    TRIAGE_LABELS_SHORT,
)

# This screen prints the diacritized complaint ("dárd", "sēna"), which the
# default Windows console codepage cannot encode. Without this the script
# crashes with UnicodeEncodeError right before showing the triage result.
make_console_safe()

# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("   ROMAN URDU EMERGENCY TRIAGE SYSTEM")
print("=" * 60)
_cli_dir = None
if "--model-dir" in sys.argv:
    _cli_dir = sys.argv[sys.argv.index("--model-dir") + 1]
MODEL_DIR, _note = resolve_model_dir(_cli_dir)
if _note:
    print(f"[note] {_note}")

print(f"Loading model from {MODEL_DIR}/ ...", end=" ")
art = load_artifacts(MODEL_DIR)
INFO = describe_model(MODEL_DIR)
print("[ok] Ready.")
print(f"  Deployed method : {INFO['method']}")
print(f"  Text features   : {INFO['basis']}"
      + (f"  ({INFO['embedding_model']}, {INFO['embedding_dim']} dims)"
         if INFO['uses_embeddings'] else ""))
print()

# ============================================================
# SAFE INPUT HELPERS
# ============================================================

def ask_float(prompt, low, high):
    while True:
        try:
            val = float(input(prompt))
            if low <= val <= high:
                return val
            print(f"  [x] Please enter a value between {low} and {high}.")
        except ValueError:
            print("  [x] Please enter a number.")


def ask_choice(prompt, options):
    """Show numbered options and return the chosen string."""
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        try:
            idx = int(input("  Enter number: "))
            if 1 <= idx <= len(options):
                return options[idx - 1]
            print(f"  [x] Enter a number between 1 and {len(options)}.")
        except ValueError:
            print("  [x] Please enter a number.")

# ============================================================
# COLLECT INPUT FROM USER
# ============================================================

def collect_patient_data():
    print("\n" + "-" * 60)
    print("  ENTER PATIENT DETAILS")
    print("-" * 60)

    complaint = input("\nChief Complaint (Roman Urdu / English):\n  > ").strip()
    if not complaint:
        complaint = "unknown"

    print()
    age          = ask_float("Age (years, 0-120):          ", 0, 120)
    heart_rate   = ask_float("Heart Rate (bpm, 20-300):    ", 20, 300)
    systolic_bp  = ask_float("Systolic BP (mmHg, 50-300):  ", 50, 300)
    diastolic_bp = ask_float("Diastolic BP (mmHg, 20-200): ", 20, 200)
    temperature  = ask_float("Temperature (C, 30-45):      ", 30, 45)
    spo2         = ask_float("SpO2 (%, 50-100):            ", 50, 100)

    print()
    gender          = ask_choice("Gender:",              list(art['le_gender'].classes_))
    mode_of_arrival = ask_choice("Mode of Arrival:",     list(art['le_mode'].classes_))
    avpu            = ask_choice("AVPU (Consciousness):", list(art['le_avpu'].classes_))
    ecg_status      = ask_choice("ECG Status:",           list(art['le_ecg'].classes_))

    return {
        "complaint": complaint,
        "age": age, "heart_rate": heart_rate,
        "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
        "temperature": temperature, "spo2": spo2,
        "gender": gender, "mode_of_arrival": mode_of_arrival,
        "avpu": avpu, "ecg_status": ecg_status,
    }

# ============================================================
# DISPLAY RESULT
# ============================================================

def show_result(p, level, confidence, proba):
    label, description = TRIAGE_LABELS_SHORT[level]

    print("\n" + "=" * 60)
    print("  TRIAGE RESULT")
    print("=" * 60)
    print(f"\n  Complaint  : {p['complaint']}")
    print(f"  Normalized : {normalize_roman_urdu(p['complaint'])}")
    print(f"\n  >>  Triage Level  : Level {level} - {label}")
    print(f"      {description}")
    print(f"\n  >>  Confidence    : {confidence * 100:.1f}%")
    print(f"\n  Probability breakdown:")
    for i, prob in enumerate(proba):
        bar = "#" * int(prob * 40)
        marker = " <<" if i == level else ""
        print(f"    L{i} [{bar:<40}] {prob * 100:5.1f}%{marker}")
    print()

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    while True:
        try:
            p = collect_patient_data()
            level, confidence, proba = predict_one(
                art, p["complaint"], p["age"], p["heart_rate"], p["systolic_bp"],
                p["diastolic_bp"], p["temperature"], p["spo2"], p["gender"],
                p["mode_of_arrival"], p["avpu"], p["ecg_status"],
            )
            show_result(p, level, confidence, proba)
        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye!")
            break

        # Ctrl-C at this prompt is outside the try above, so it used to escape
        # as an uncaught KeyboardInterrupt traceback instead of exiting cleanly.
        try:
            again = input("Predict another patient? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting. Goodbye!")
            break
        if again != 'y':
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    main()
