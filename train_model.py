import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load merged dataset
df = pd.read_csv("merged_dataset.csv")

# Drop ID if it exists
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Outcome"])
y = df["Outcome"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "patient_triage_model.pkl")
print("✅ Model trained and saved as patient_triage_model.pkl")
