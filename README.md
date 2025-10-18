# MedGemma-Powered Digital Health Platform: Intelligent Rare Disease Diagnosis and Triage

## 1. Project Overview

This platform addresses two core challenges in clinical workflow: **delayed diagnosis for rare diseases** and **inefficient patient triage** in high-volume settings.  
We propose a novel, multimodal, dual-model architecture to handle both problems simultaneously.

The system uses a **specialized, fine-tuned MedGemma Large Multimodal Model (LMM)** for complex, multimodal diagnostic reasoning (text + image), and a **classical Machine Learning (ML) model** for rapid, operational patient triage prediction.

### Key Contributions

1. **Agentic Orchestration Framework:**  
   A unified system that intelligently routes requests to the optimal AI component (Triage ML vs. MedGemma LMM) based on the input type and complexity.

2. **Specialized MedGemma Tuning:**  
   Implementation of a Parameter-Efficient Fine-Tuning (PEFT) method (e.g., LoRA) on MedGemma, specifically targeting a class of rare diseases (e.g., specific genetic or dermatological conditions) for enhanced diagnostic accuracy.

3. **Real-time Dual-Task Validation:**  
   A comprehensive evaluation methodology for both the precision-focused LMM and the speed-focused ML model using distinct clinical metrics.

---

## 2. Technical Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Multimodal LLM** | MedGemma (Fine-tuned) | Rare Disease Differential Diagnosis & Clinical Reasoning. |
| **Triage Model** | Scikit-learn / TensorFlow (e.g., Random Forest) | High-speed prediction for Triage Urgency/Priority. |
| **API Backend** | FastAPI | High-performance, asynchronous REST API to serve both models. |
| **LLM Orchestration** | LangChain / Custom Python Agents | Manages prompt injection, RAG integration, and multi-step reasoning. |
| **Frontend/Demo** | Streamlit / Gradio | Interactive web interface for end-user testing and presentation. |
| **Dependencies** | Python 3.10+, PyTorch, Hugging Face `transformers` | Core libraries for ML and LLM operations. |

---

## 3. Project Structure

A clean, modular structure is essential for an ML/LLM project of this complexity.

```bash
medgemma-health-platform/
├── app/
│ ├── api/
│ │ ├── init.py
│ │ ├── triage_router.py # FastAPI routes for the Triage ML model
│ │ └── diagnosis_router.py # FastAPI routes for the MedGemma Agent
│ ├── core/
│ │ ├── config.py # Environment variables and model paths
│ │ ├── logging.py
│ │ └── exceptions.py
│ ├── services/
│ │ ├── llm_agent.py # Logic for MedGemma prompting, RAG, and image processing
│ │ └── triage_service.py # Logic for loading/running the Triage ML model
│ └── main.py # FastAPI entry point (combines routers)
├── data/
│ ├── raw/ # Original, untouched data (e.g., clinical notes, X-rays)
│ ├── processed/ # Cleaned, de-identified data for ML/LLM training
│ └── knowledge_base/ # RAG documents (e.g., Rare Disease literature PDFs)
├── models/
│ ├── triage_model.pkl # Saved Triage ML model (e.g., Random Forest)
│ └── medgemma_lora_weights/ # LoRA/PEFT weights for MedGemma fine-tuning
├── notebooks/
│ ├── 1.0-EDA.ipynb
│ ├── 2.0-Triage_Model_Training.ipynb
│ └── 3.0-MedGemma_FineTuning.ipynb # Detailed steps for LoRA/PEFT
├── tests/
│ ├── test_api.py # Unit tests for FastAPI endpoints
│ └── test_models.py # Unit tests for core model prediction logic
├── research_paper/
│ ├── paper.docx / paper.tex # The final research paper document
│ └── figures/ # Diagrams, charts, and architecture visuals
├── requirements.txt # Project dependencies (pip freeze > requirements.txt)
├── Dockerfile # For containerizing the application
└── README.md # This file
```

---

## 4. Getting Started

### 4.1. Prerequisites

- Python 3.10 or higher  
- GPU access (e.g., Google Colab Enterprise, or local setup with NVIDIA CUDA) is **highly recommended** for fine-tuning and running MedGemma 27B  
- A Hugging Face or Google Cloud API key if using cloud-based MedGemma APIs or hosted models

### 4.2. Installation

1. **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd medgemma-health-platform
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 4.3. Model Setup

1. **Triage Model:**  
   The trained ML model (`triage_model.pkl`) must be placed in the `models/` directory.  
   Run `notebooks/2.0-Triage_Model_Training.ipynb` to train it if missing.

2. **MedGemma:**  
   - **Weights:** Download fine-tuned LoRA weights into `models/medgemma_lora_weights/`  
   - **RAG:** Place the clinical/rare disease PDF or text files into `data/knowledge_base/`  
     (`app/services/llm_agent.py` will handle embedding and retrieval.)

### 4.4. Running the API

1. **Set Environment Variables:**  
   Create a `.env` file in the root directory and add necessary keys (e.g., API keys, model path configs).

2. **Start the FastAPI Server:**
    ```bash
    uvicorn app.main:app --reload
    ```
    API docs available at **`http://127.0.0.1:8000/docs`**

---

## 5. Paper Submission Checklist

| Task | Status | Notes |
| :--- | :--- | :--- |
| **Data Card** | [ ] | Document the size, source, and PHI de-identification status of the dataset. |
| **Reproducibility** | [ ] | Ensure `requirements.txt` is complete and all setup steps are clear. |
| **LLM Tuning Details** | [ ] | Explicitly state MedGemma version, LoRA rank (r), and training data size. |
| **Evaluation Metrics** | [ ] | Report **Accuracy/F1-score** for Triage model AND **Top-K Accuracy/Clinical Utility** for MedGemma. |
| **Architecture Diagram** | [ ] | Include a clear figure showing the flow from input → dual-model output. |

---

## 6. Contact and Authorship

| Name | Role | Contact |
| :--- | :--- | :--- |
| **Muawiya Amir** | Lead Developer — Responsible for model development and implementation | contactmuawia@gmail.com |
| **Wasiq Siddiqui** | Research Lead — Responsible for literature review and research paper preparation | [Your Email/LinkedIn] |

---

### 🧩 Next Step
Start by populating `app/core/config.py` and creating the initial files in the `app/api/` and `app/services/` directories.
---
