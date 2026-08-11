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
#   python predict_batch.py my_patients.csv --model-dir triage_model
#
# By default this runs the DEPLOYED model in triage_model_embedding/ (the
# offline sentence-embedding pipeline). Pass --model-dir triage_model to
# run the older dictionary + Bag-of-Words model instead. Whichever runs is
# printed at startup, so the method behind a result sheet is never a guess.
#
# CSV files are read with encoding detection (utf-8 -> utf-8-sig -> cp1252
# -> latin-1), so a sheet exported from Excel on Windows no longer fails
# with UnicodeDecodeError.
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
    describe_model,
    load_artifacts,
    predict_dataframe,
    read_table,
    resolve_model_dir,
    REQUIRED_INPUT_COLUMNS,
    TRIAGE_LABELS,
)

DEFAULT_INPUT = "batch_input_template.xlsx"

# read_table now lives in triage_pipeline.py so this script and the GUI's
# batch tab share ONE reader. The local copy here used a bare
# pd.read_csv(path), which assumes UTF-8 and died with UnicodeDecodeError
# on any CSV exported from Excel on a Windows machine.


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
    argv = [a for a in sys.argv[1:]]
    model_dir = None
    if "--model-dir" in argv:
        i = argv.index("--model-dir")
        model_dir = argv[i + 1]
        del argv[i:i + 2]

    in_path = argv[0] if argv else DEFAULT_INPUT

    if not os.path.exists(in_path):
        print(f"[error] Input file not found: {in_path}")
        print(f"        Provide a CSV/Excel file, e.g.:  python predict_batch.py my_patients.xlsx")
        sys.exit(1)

    if len(argv) > 1:
        out_base = os.path.splitext(argv[1])[0]
    else:
        out_base = os.path.splitext(in_path)[0] + "_predictions"

    # ---- load model ----
    model_dir, note = resolve_model_dir(model_dir)
    if note:
        print(f"[note] {note}")
    print(f"Loading model and encoders from '{model_dir}/'...")
    art = load_artifacts(model_dir)
    info = describe_model(model_dir)
    print(f"[ok] Model ready.")
    print(f"     Deployed method : {info['method']}")
    print(f"     Text features   : {info['basis']}"
          + (f"  ({info['embedding_model']}, {info['embedding_dim']} dims)"
             if info['uses_embeddings'] else ""))
    print()

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
