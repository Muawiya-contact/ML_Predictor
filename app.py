from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import pandas as pd
import joblib
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from train_model import train_and_save_model, merge_and_save_dataset, get_feature_metadata
from predict import predict_patient
from gpt_helper import get_gpt_explanation

app = Flask(__name__, static_folder="static")
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = "patient_triage_model.pkl"
DATASET_PATH = "merged_dataset.csv"
FEATURE_META_PATH = "model_features.json"
HISTORY_PATH = "history.json"

# Load history
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history[-10:], f)  # keep last 10

@app.route("/", methods=["GET"])
def home():
    features = get_feature_metadata(FEATURE_META_PATH)
    return render_template("index.html", features=features, result=None, history=load_history())

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename.endswith(".csv"):
        return "Invalid file", 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    # Merge and retrain
    merge_and_save_dataset(filepath, DATASET_PATH)
    train_and_save_model(DATASET_PATH, MODEL_PATH, FEATURE_META_PATH)
    return redirect(url_for("home"))

@app.route("/predict", methods=["POST"])
def predict():
    features = get_feature_metadata(FEATURE_META_PATH)
    if request.is_json:
        data = request.get_json(force=True)
    else:
        data = request.form.to_dict()
    # Fill missing with mean/default
    df = pd.read_csv(DATASET_PATH)
    input_data = {}
    for feat in features:
        val = data.get(feat)
        if val is None or val == "":
            # Use mean if numeric, else 0
            if pd.api.types.is_numeric_dtype(df[feat]):
                input_data[feat] = float(df[feat].mean())
            else:
                input_data[feat] = 0
        else:
            input_data[feat] = float(val)
    prediction = predict_patient(input_data, MODEL_PATH, features)
    explanation = get_gpt_explanation(prediction, input_data)
    # Save to history
    history = load_history()
    history.append({
        "timestamp": datetime.now().isoformat(),
        "inputs": input_data,
        "prediction": prediction,
        "explanation": explanation
    })
    save_history(history)
    return jsonify({
        "prediction": prediction,
        "explanation": explanation
    })

@app.route("/history", methods=["GET"])
def history():
    return jsonify(load_history())

@app.route("/retrain", methods=["POST"])
def retrain():
    train_and_save_model(DATASET_PATH, MODEL_PATH, FEATURE_META_PATH)
    return "Retrained", 200

if __name__ == "__main__":
    print("🚀 Flask app running at http://127.0.0.1:5000")
    app.run(debug=True)
