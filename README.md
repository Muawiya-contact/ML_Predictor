# 🏥 MedGemma-Powered Digital Health Platform

## 📖 Project Overview

This repository hosts a comprehensive **Digital Health AI Platform** that leverages **Google’s MedGemma** foundation models for advanced **clinical decision support**.  
The platform is built around two complementary components:

### 1. **Diagnosys: Rare Disease Navigator AI**
A **multimodal AI agent** that synthesizes unstructured **clinical notes and medical images** to generate evidence-based **Rare Disease Differential Diagnoses (RD-DDx)**.

### 2. **Patient Triage Prediction System**
A **classical Machine Learning model** for **initial patient assessment**, predicting patient outcomes (e.g., *ICU, OT, Ward*) using **structured clinical and lab parameters**.

Together, these systems provide **end-to-end diagnostic reasoning and operational triage support** — bridging ML efficiency with LLM-based interpretability.

---

## 👨‍💻 Developed By
**Muawiya Amir** — AI Student, NFC IET Multan  
**Research Collaboration:** Wasiq Siddiqui (BMT)

---

## 🚀 Getting Started

### 🔧 Prerequisites
- **Google Cloud Vertex AI Access** — for deploying and serving MedGemma models  
- **Hugging Face Credentials** — for accessing MedGemma model weights  
- **Secure Datasets** — both:
  - Rare-disease data (for LLM fine-tuning)  
  - Structured clinical/lab data (for ML model training)

---

### ⚙️ Installation & Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/YourStartup/diagnosys-medgemma-ai.git
cd diagnosys-medgemma-ai
```
#### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```
### 3. Configure Environment Variables

Create a `.env` file in the root directory and populate it with:
```bash
GCP_PROJECT_ID=your_project_id
VERTEX_ENDPOINT_URL=https://your-model-endpoint
HUGGINGFACE_TOKEN=your_hf_token
```
### 📂 Project Structure

The architecture separates the **MedGemma-based diagnostic system** and the **classical ML triage predictor** for modular development.
```bash
diagnosys-medgemma-ai/
│
├── data/
│   ├── llm_tuning/           # Rare Disease data for MedGemma (text/images)
│   └── ml_training/          # Structured data for Triage ML model (CSV/Parquet)
│
├── diagnosys/                # Multimodal LLM System (Rare Disease Navigator)
│   ├── models/               # MedGemma fine-tuning code and adapter weights
│   ├── agents/               # Core agentic orchestration logic
│   └── knowledge_base/       # RAG documents for clinical reasoning
│
├── triage_ml/                # Classical ML System (Patient Triage Prediction)
│   ├── notebooks/            # Exploratory Data Analysis & feature engineering
│   ├── prediction_model/     # Model code and serialized files (e.g., model.pkl)
│   └── scripts/              # Training and evaluation scripts
│
├── app/                      # Unified API & Frontend Layer
│   ├── api/                  # Flask/FastAPI routes for both components
│   ├── webapp/               # Web interface for clinicians
│   │   ├── public/           # Static assets (CSS, JS)
│   │   └── src/              # Source code (HTML templates, React/Vue optional)
│   └── utils/                # Helper functions (validation, cleaning, etc.)
│
├── infrastructure/           # Deployment configs (Terraform, CI/CD)
├── .env.example
├── requirements.txt
└── README.md
```
## 🖥 User Interface (UI)

The unified **Diagnosys UI** provides clinicians with two main panels:

| **Panel** | **Function** |
|------------|--------------|
| **Triage View (Top Panel)** | Quick entry of structured labs/vitals → Predicts immediate disposition *(ICU / OT / Ward)* |
| **Diagnostic View (Main Panel)** | Input for EMR text + medical images → Generates rare disease differential diagnoses and explanations |

*(Image preview placeholder: `assets/ui_overview.png`)*

---

## 🛠 Component Details

### 🧬 1. Diagnosys: Rare Disease Navigator AI (LLM)

- **Model:** MedGemma-27B-Multimodal  
- **Deployment:** Google Vertex AI Endpoint  
- **Fine-tuning:** LoRA adapters using curated rare-disease datasets in `data/llm_tuning/`

**Pipeline:**
1. Extract findings from clinical text & images  
2. Query knowledge base (RAG) for relevant literature  
3. Generate structured reasoning and RD-DDx explanation  

---

### 🧠 2. Patient Triage Prediction System (ML)

- **Model Type:** Gradient Boosting / Random Forest / MLP (scikit-learn or TensorFlow)  
- **Input:** Structured clinical parameters *(vitals, lab values, symptoms)*  
- **Output:** Predicted care destination *(ICU / OT / Ward)*  

**Features:**
- Automatic duplicate removal & column alignment for custom datasets  
- Re-trainable model on new datasets  
- Integrated GPT/MedGemma reasoning for interpretability  

---

## 🚢 Deployment

We use **Terraform** for reproducible and scalable deployment across **Google Cloud**.

```bash
cd infrastructure/terraform/
terraform init
terraform apply
```
## Deployed Components

+ 🧠 MedGemma endpoint on Vertex AI

+ ⚙️ ML model hosted on Cloud Run

+ 🌐 Unified API Gateway connecting both services

---
## 📚 Research & Credits

+ Developed by: Muawiya Amir (AI Student, NFC IET Multan)

+ In Collaboration With: Wasiq Siddiqui (BMT — Biomedical Engineering)

+ Affiliation: BMT-201 Research Series — Explainable AI for Healthcare

---
## 🧩 Future Enhancements

 + 🩻 Add image-based diagnosis support (X-ray, MRI integration)

  🧬 Fine-tune MedGemma on local hospital data

 + 🔒 Add Federated Learning for data privacy

 + 📊 Build interactive visual dashboard for predictions

---
## 💡 Summary

BMT-201 represents a fusion of structured ML and LLM reasoning, enabling transparent, explainable AI for healthcare.
This approach aligns with modern clinical AI trends — balancing automation with accountability, and accuracy with explainability.

> "Where machine intelligence meets clinical empathy."
> — BMT Research Initiative 2025
>
> ---
