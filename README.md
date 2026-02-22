# MedGemma-Sentinel: AI-Driven ICU Sepsis Monitor

**MedGemma-Sentinel** is an intelligent clinical agent designed to bridge the gap between Biomedical Technology (BMT) and Artificial Intelligence. It monitors real-time ICU patient vitals to provide early warnings for Sepsis using a bilingual generative-reasoning architecture.

---

## Project Overview

Early detection of Sepsis is critical in ICU environments. This project implements an Agentic AI system that analyzes patient data through:

### Clinical Reasoning  
Utilizes google/gemma-2-2b-it (Gemma 2 architecture) to interpret complex medical trends using structured reasoning.

### Bilingual Context Understanding  
Native support for clinical notes in English and Urdu (e.g., "saans ka masla"), tailored for the Pakistani healthcare landscape.

### Interactive GUI  
A Streamlit-based interface allows doctors and nurses to input patient data and receive instant, explainable triage results.

---

## Monitored Parameters

The agent evaluates clinical metrics against SIRS (Systemic Inflammatory Response Syndrome) and SOFA criteria:

- **Heart Rate (HR):** Detects Tachycardia (> 90 bpm)
- **Blood Pressure (BP):** Monitors Hypotension (< 90/60 mmHg)
- **Temperature:**  
  - Fever (> 100.4°F)  
  - Hypothermia (< 96.8°F)
- **WBC Count:** Assesses immune response abnormalities
- **Qualitative Notes:** Parses linguistic cues for:
  - Respiratory distress  
  - Altered mental status  
  - Systemic weakness  

---

## Technical Architecture

- **Core Model:** google/gemma-2-2b-it  
- **Precision:** 16-bit Floating Point (bfloat16) for GPU efficiency  
- **Reasoning Method:** Chain-of-Thought (CoT) prompting — forces the AI to explain its logic before producing a status  
- **Frontend Layer:** Streamlit Web Framework served via secure port-forwarding  
- **Fail-Safe Mechanism:** Logic-gate layer ensures 100% monitoring uptime even if LLM inference experiences latency  

---

## Deployment Results (Bilingual Capabilities)

| Patient Input | Clinical Note (English/Urdu) | AI Status | Reasoning Excerpt |
|--------------|------------------------------|------------|------------------|
| HR 75, Temp 98.6 | Routine recovery | STABLE | Vitals are within normal physiological ranges. |
| HR 130, BP 85/50 | Saan lenay mein masla hai | CRITICAL | Tachycardia + Hypotension + Respiratory Distress detected. |
| HR 110, Temp 101.5 | Bohot kamzoori hai | ELEVATED | SIRS positive (HR + Temp). Weakness indicates systemic infection. |

---

## Future Roadmap

- [ ] MIMIC-IV Integration: Validate agent logic against 40,000+ real ICU cases  
- [ ] Multi-Modal Support: Incorporate Chest X-Rays and ECG waveforms  
- [ ] Local Deployment: Optimize quantized execution for low-resource rural clinics in Pakistan  

---

## Ethical Considerations

This project follows CITI/RCR research training guidelines.

- Designed strictly as a Clinical Decision Support Tool  
- Does not replace human medical judgment  
- Ensures data privacy using secure, localizable model weights  
- No external cloud storage required  
