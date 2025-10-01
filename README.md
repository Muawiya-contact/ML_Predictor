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
