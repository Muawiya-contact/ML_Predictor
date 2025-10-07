import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import json
import os

def merge_and_save_dataset(new_csv, merged_csv):
    if os.path.exists(merged_csv):
        base = pd.read_csv(merged_csv)
        new = pd.read_csv(new_csv)
        df = pd.concat([base, new], ignore_index=True)
        df = df.drop_duplicates()
    else:
        df = pd.read_csv(new_csv)
    df.to_csv(merged_csv, index=False)

def train_and_save_model(dataset_csv, model_path, feature_meta_path):
    df = pd.read_csv(dataset_csv)
    if "Outcome" not in df.columns:
        raise Exception("No Outcome column in dataset")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    # Detect numeric/categorical
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = [col for col in X.columns if col not in num_cols]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    X_proc = preprocessor.fit_transform(X)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_proc, y)
    joblib.dump((model, preprocessor), model_path)
    # Save feature order
    with open(feature_meta_path, "w") as f:
        json.dump(list(X.columns), f)

def get_feature_metadata(feature_meta_path):
    if os.path.exists(feature_meta_path):
        with open(feature_meta_path) as f:
            return json.load(f)
    return []

if __name__ == "__main__":
    merge_and_save_dataset("CardiacPatientData.csv", "merged_dataset.csv")
    train_and_save_model("merged_dataset.csv", "patient_triage_model.pkl", "model_features.json")
