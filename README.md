
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
# Patient Triage Prediction System

A Machine Learning-based system to predict patient outcomes (e.g., ICU, OT, Ward) from clinical and lab parameters. The system also integrates GPT-based suggestions for patient care recommendations.

---

## 🗂 Project Structure

```txt
BMT/
│── templates/
│ └── index.html # Frontend HTML form
│
│── static/
│ └── styles.css # External CSS for styling
│
│── .env # Environment variables (e.g., API keys)
│── .gitignore
│── app.py # Flask app serving the web interface and prediction API
│── gpt_helper.py # GPT helper functions for explanation/suggestions
│── main.py # Optional entry point for testing or future features
│── prepare_data.py # Data cleaning, merging, preprocessing
│── train_model.py # Train ML model and save .pkl file
│── predict.py # Predict function for single patient input
│── merged_dataset.csv # Merged dataset from multiple sources
│── patient_triage_model.pkl # Trained RandomForest model
│── balanced_triage_dataset.csv # Optional balanced dataset
│── 30 patients dataset.csv # Sample dataset for testing

```

---

## ⚡ Features (Planned / Skeleton)

- Predict patient outcome (ICU, OT, Ward) from clinical parameters
- Preprocessing of multiple datasets into a merged dataset
- RandomForest ML model for prediction
- GPT-based suggestions for patient care (via API)
- Web interface with HTML form for doctors to input patient data
- Responsive and accessible form design (labels, placeholders, title attributes)

---

## 📝 How to Run

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Flask app
```bash
python app.py
```

3. Open the web interface
```
http://127.0.0.1:5000/

```
4. Enter patient parameters and click ***Predict*** to get outcome + GPT suggestion.

----

### 🔧 Next Steps / TODO

+ Add more datasets and merge intelligently

+ Improve ML model accuracy with hyperparameter tuning

+ Enhance GPT suggestions based on patient context

+ Add user authentication for doctors

+ Improve web interface UX (responsive design, error handling)

+ Dockerize the project for deployment

----
