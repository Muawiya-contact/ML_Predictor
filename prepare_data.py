import pandas as pd

def prepare_data():
    # Load datasets
    df1 = pd.read_csv("CardiacPatientData.csv")
    df2 = pd.read_csv("balanced_triage_dataset.csv")

    # Rename dataset2 columns
    df2.rename(columns={
        "Sex": "Gender",
        "Temp_C": "BT",
        "HeartRate": "HR",
        "RespRate": "RR",
        "SystolicBP": "SBP",
        "DiastolicBP": "DBP",
        "TriageLevel": "Outcome"
    }, inplace=True)

    # Convert Gender: M=1, F=0
    df2["Gender"] = df2["Gender"].map({"M": 1, "F": 0})

    # Add missing columns from df1 to df2
    for col in df1.columns:
        if col not in df2.columns:
            df2[col] = 0

    # Align order
    df2 = df2[df1.columns]

    # Merge
    merged = pd.concat([df1, df2], ignore_index=True)
    merged.to_csv("merged_dataset.csv", index=False)
    print("✅ Merged dataset saved as merged_dataset.csv")

if __name__ == "__main__":
    prepare_data()
