# 🏥 MedGemma-Sentinel: AI-Driven ICU Sepsis Monitor

**MedGemma-Sentinel** is an intelligent clinical agent designed to bridge the gap between Biomedical Technology (BMT) and Artificial Intelligence. It monitors real-time ICU patient vitals to provide early warnings for Sepsis using a dual-path architecture.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Muawiya-contact/ML_Predictor/blob/main/MedGemma.ipynb)

## 🌟 Project Overview
Early detection of Sepsis is critical in ICU environments. This project implements an **Agentic AI** that analyzes patient data through:
1. **Clinical Reasoning:** Utilizing the Gemma-2-2b-it model to interpret medical trends.
2. **Safety Fallback:** A deterministic rule-based system based on SIRS (Systemic Inflammatory Response Syndrome) and SOFA criteria.



## 🛠️ Monitored Parameters
The agent evaluates the following clinical metrics to assess patient stability:
- **Heart Rate (HR):** Identification of Tachycardia (>100 bpm).
- **Blood Pressure (BP):** Monitoring for Hypotension (e.g., <90/60 mmHg).
- **Temperature:** Tracking Fever (>101°F) or Hypothermia (<96.8°F).
- **WBC Count:** Analyzing immune response markers.

## 🏗️ Technical Architecture
- **Model:** `google/gemma-2-2b-it` (Gemma 2 Architecture).
- **Precision:** 16-bit Floating Point (`bfloat16`) for GPU efficiency.
- **Framework:** Hugging Face `transformers` & `torch`.
- **Fail-Safe:** Implemented a `try-except` logic that defaults to a Rule-Based Expert System if LLM inference is unavailable, ensuring 100% monitoring uptime.

## 📊 Sample Results
| Patient Case | Status | Reasoning |
| :--- | :--- | :--- |
| **Case A** | ✅ STABLE | Vitals within normal physiological ranges. |
| **Case B** | ⚠️ ELEVATED | Rising HR and borderline BP; early intervention suggested. |
| **Case C** | 🚨 CRITICAL | Tachycardia + Hypotension detected. High risk of Septic Shock. |

## 🚀 Future Roadmap
- [ ] **MIMIC-IV Integration:** Validate agent logic against 40,000+ real ICU cases.
- [ ] **Longitudinal Analysis:** Track vitals over 6-12 hour windows for trend prediction.
- [ ] **Multimodal Inputs:** Incorporate lab results (Lactate, Creatinine) for more accurate SOFA scoring.

## ⚖️ Ethical Considerations
This project was developed following **CITI/RCR training guidelines**. It is designed as a **Decision Support Tool** to assist clinicians, not to replace human medical judgment. Data privacy is maintained by utilizing secure, localizable model weights.