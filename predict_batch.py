# predict_batch.py
# ============================================================
# BATCH TRIAGE PREDICTION FROM A FILE
# ============================================================
# Predict the triage level for many patients at once (e.g. 100
# at a time) by uploading a single Excel (.xlsx) or CSV file.
#
# USAGE
# -----
#   python predict_batch.py                       # uses batch_input_template.xlsx
#   python predict_batch.py my_patients.xlsx
#   python predict_batch.py my_patients.csv
#   python predict_batch.py my_patients.xlsx  results.xlsx
#
# INPUT FILE COLUMNS (header row required, any order):
#   Complaint_Text, Age, Gender, Mode_of_Arrival, Heart_Rate,
#   Systolic_BP, Diastolic_BP, Temperature, SpO2, AVPU, ECG_Status
#
# Missing numbers are filled with the training average and unknown
# category values fall back safely (a note is written per row), so
# a slightly messy file will still produce a full result sheet.
#
# OUTPUT
# ------
#   * Saves <inputname>_predictions.xlsx  (and .csv) next to the input
#   * Adds columns: Predicted_Triage_Level (1-4), Predicted_Label,
#     Confidence, P_L0..P_L3, Notes
#   * Prints a summary table + triage-level counts to the terminal
# ============================================================

import os
import sys
import pandas as pd

from triage_pipeline import (
    load_artifacts,
    predict_dataframe,
    REQUIRED_INPUT_COLUMNS,
    TRIAGE_LABELS,
)

DEFAULT_INPUT = "batch_input_template.xlsx"


def read_table(path):
    """Read a CSV or Excel file into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type '{ext}'. Use .xlsx or .csv")


def write_table(df, base_path_no_ext):
    """Write results to both .xlsx and .csv. Returns list of written paths."""
    written = []
    csv_path = base_path_no_ext + ".csv"
    df.to_csv(csv_path, index=False)
    written.append(csv_path)
    try:
        xlsx_path = base_path_no_ext + ".xlsx"
        df.to_excel(xlsx_path, index=False)
        written.append(xlsx_path)
    except Exception as e:
        print(f"  (Excel output skipped: {e})")
    return written


def main():
    # ---- resolve input / output paths ----
    in_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT

    if not os.path.exists(in_path):
        print(f"[error] Input file not found: {in_path}")
        print(f"        Provide a CSV/Excel file, e.g.:  python predict_batch.py my_patients.xlsx")
        sys.exit(1)

    if len(sys.argv) > 2:
        out_base = os.path.splitext(sys.argv[2])[0]
    else:
        out_base = os.path.splitext(in_path)[0] + "_predictions"

    # ---- load model ----
    print("Loading model and encoders...")
    art = load_artifacts("triage_model")
    print("[ok] Model ready.\n")

    # ---- read patients ----
    print(f"Reading patients from: {in_path}")
    df = read_table(in_path)
    print(f"[ok] {len(df)} patient rows found.")

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        print("\n[warning] These expected columns are missing and will be "
              "treated as blank (filled with safe defaults):")
        for c in missing:
            print(f"            - {c}")
        print()

    # ---- predict ----
    print("Predicting triage levels...")
    results, _ = predict_dataframe(art, df)

    # ---- save ----
    written = write_table(results, out_base)
    print("\n[ok] Predictions written to:")
    for w in written:
        print(f"        {w}")

    # ---- terminal summary ----
    print("\n" + "=" * 78)
    print("BATCH TRIAGE SUMMARY")
    print("=" * 78)

    counts = results['Predicted_Triage_Level'].value_counts().sort_index()
    label_names = {1: "EMERGENCY", 2: "URGENT", 3: "STANDARD", 4: "NON-URGENT"}
    for lvl in [1, 2, 3, 4]:
        n = int(counts.get(lvl, 0))
        bar = "#" * n
        print(f"  Level {lvl} ({label_names[lvl]:<10}) : {n:>4}  {bar}")
    print(f"  {'TOTAL':<22} : {len(results):>4}")

    # ---- preview first rows ----
    print("\n" + "-" * 78)
    print("PREVIEW (first 10 rows)")
    print("-" * 78)
    preview_cols = ['Complaint_Text', 'Age', 'Predicted_Triage_Level',
                    'Predicted_Label', 'Confidence']
    preview_cols = [c for c in preview_cols if c in results.columns]
    with pd.option_context('display.max_colwidth', 40, 'display.width', 120):
        print(results[preview_cols].head(10).to_string(index=False))

    notes = results[results['Notes'].astype(str).str.len() > 0]
    if len(notes):
        print(f"\n[note] {len(notes)} row(s) had missing/unknown values that were "
              f"auto-filled. See the 'Notes' column in the output file.")

    print("\nDone.")


if __name__ == "__main__":
    main()
