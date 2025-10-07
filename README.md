# Adaptive Medical Triage Prediction System

A Flask-based intelligent prediction system for medical triage, featuring:

- **Automatic model training** from any uploaded medical dataset (CSV)
- **Dynamic web form generation** based on detected features
- **Dataset merging** with duplicate removal
- **Outcome prediction** (e.g., admission place, triage level) even with partial inputs
- **AI-generated explanations** (via GPT)
- **Prediction history** and on-demand retraining

---

## 👨‍💻 Developed By

**Muawiya Amir** — AI Student at NFC IET Multan  
**Research Collaboration:** Wasiq Siddiqui (BMT)

---

## 🚀 Features

- **Upload Dataset:** Upload any CSV; system merges and retrains automatically.
- **Dynamic Input Form:** Web form updates to match model features.
- **Prediction:** Handles missing values, predicts outcomes, and provides GPT-based suggestions.
- **History:** Stores and displays last 10 predictions.
- **Retrain:** Retrain model on demand.

---

## 🧠 Tech Stack

- **Frontend:** HTML, CSS, Jinja2
- **Backend:** Flask
- **ML:** scikit-learn, pandas, joblib
- **AI Explanation:** OpenAI GPT (optional)
- **Storage:** CSV

---

## ⚙️ Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   python app.py
   ```

3. **Open in your browser:**  
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

4. **Workflow:**
   - Upload a dataset (CSV)
   - Fill in the dynamically generated form
   - Click "Predict" to get triage results and AI suggestions

---

## 📁 File Structure

```
├── app.py                # Main Flask app
├── templates/
│   └── index.html        # Dynamic frontend
├── static/
│   └── style.css         # UI styling
├── train_model.py        # Dataset merging & model training
├── predict.py            # Model loading & prediction
├── gpt_helper.py         # GPT-based explanations
└── requirements.txt      # Dependencies
```

---

## 🧾 Example Dataset Columns

`SBP, DBP, HR, RR, BT, SpO2, Age, Gender, GCS, Na, K, Cl, Urea, Ceratinine, Alcoholic, Smoke, FHCD, TriageScore, Outcome`

---

## 📚 Research & Credits

Developed by **Muawiya Amir** (AI Student, NFC IET Multan)  
In collaboration with **Wasiq Siddiqui** (BMT)