from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from gpt_helper import explain_prediction

app = Flask(__name__)

# Load model
model = joblib.load("patient_triage_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    explanation = explain_prediction(prediction)

    return jsonify({
        "prediction": str(prediction),
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(debug=True)
