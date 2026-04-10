# 🏥 DiabetesGuard AI — Multi-Agent Patient Triage & Readmission Risk Dashboard
**A Generative AI Solution for Predicting and Preventing Diabetes Hospital Re-admissions**

DiabetesGuard AI addresses a critical challenge in healthcare: identifying diabetic patients at high risk of being readmitted within 30 days of discharge. This system processes patient data through a specialised **multi-agent Azure OpenAI pipeline** to categorise urgency, explain predictions, and generate personalised discharge plans — trained on two real clinical datasets.

---

## 🎓 Academic & Professional Context

| | |
|---|---|
| **Author** | Temitope Adereni |
| **Programme** | Generative AI & Data Science Course |
| **Internship** | DataraFlow Data Science Internship Programme |
| **ID** | DF2025-036 |
| **Supervisor** | Winner Emeto (DataraFlow) |
| **Focus** | Multi-Agent Orchestration, Azure OpenAI Integration, Healthcare ML |

---

## 🔗 Live Demo

> 🚀 **[(https://diabetes-guard-ai.streamlit.app/)](#)** 

---

## 🔬 Research Foundation

This application is built on the research paper:

> **Predictive Modeling: Utilizing Machine Learning to Forecast Diabetes Hospital Re-admissions**
> *Temitope Adereni — DataraFlow Internship Programme, DF2025-036*

Hospital readmissions cost the US healthcare system over **$17 billion annually**. Among chronic conditions, diabetes mellitus is one of the leading drivers — its complex management creates frequent treatment gaps that result in rapid return to hospital. This project builds a clinically meaningful, interpretable prediction system using two real-world datasets.

---

## 🏗️ Technical Architecture — The Multi-Agent Pipeline

Unlike standard chatbots, DiabetesGuard AI uses a **Multi-Agent Orchestration** pattern. When a clinician submits patient data, the system triggers three specialised agents via **Azure OpenAI (GPT-4o)**:

### 🔴 Agent 1 — RiskAssessor
- **Role:** Emergency Triage
- **Input:** Utilisation features (Prior Inpatient Visits, Length of Stay, Emergency Visits), Clinical variables (HbA1c, Diagnoses, Medications)
- **Output:** Urgency Category — 🔴 High / 🟡 Moderate / 🟢 Low + risk percentage score
- **Grounded in:** LR model AUC = 0.624 (UCI), 0.644 (Secondary)

### 🔵 Agent 2 — ExplainerAgent
- **Role:** Explainable AI / Feature Attribution
- **Input:** Full patient profile + risk score
- **Output:** 3 specific bullet points explaining *which features drive risk and why*
- **Grounded in:** H2 finding — utilisation features improved AUC from 0.529 → 0.629 (+19%)

### 🟢 Agent 3 — DischargeAdvisor
- **Role:** Clinical Discharge Planning
- **Input:** Patient profile + risk level
- **Output:** 4 personalised, empathetic discharge recommendations to reduce readmission
- **Grounded in:** Clinical literature + dataset-specific patient characteristics

### 🟣 Agent 4 — ResearchAssistant
- **Role:** Research Q&A Chat
- **Input:** Any question about the research, methodology, or datasets
- **Output:** Detailed answers referencing real metrics from both datasets

---

## 📊 Datasets

### Dataset 1 — UCI Diabetes 130-US Hospitals (1999–2008)
| Property | Value |
|---|---|
| Rows | 101,766 encounters |
| Features | 42 columns |
| Target | `readmitted` — `<30` days = High Risk (1), else = 0 |
| Class balance | NO=53.9%, >30d=34.9%, **<30d=11.2%** (heavily imbalanced) |
| Source | [UCI ML Repository](https://doi.org/10.24432/C5230J) |

**Key features:** `race`, `gender`, `age`, `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `diag_1–3`, `number_diagnoses`, `A1Cresult`, 23 medication columns, `change`, `diabetesMed`

### Dataset 2 — Secondary Hospital Readmissions
| Property | Value |
|---|---|
| Rows | 25,000 encounters |
| Features | 13 columns |
| Target | `readmitted` — `yes`/`no` (binary) |
| Class balance | No=52.98%, **Yes=47.02%** (near-balanced) |

**Key features:** `age`, `time_in_hospital`, `n_lab_procedures`, `n_procedures`, `n_medications`, `n_outpatient`, `n_inpatient`, `n_emergency`, `glucose_test`, `A1Ctest`, `change`, `diabetes_med`

---

## 📈 Research Hypotheses & Results

### H1 — Model Benchmarking
> *"Random Forest will outperform Logistic Regression in AUC and F1."*

| Model | Dataset | AUC | F1 |
|---|---|---|---|
| Logistic Regression | UCI | **0.624** | 0.028 |
| Random Forest | UCI | 0.609 | 0.012 |
| Logistic Regression | Secondary | **0.644** | 0.496 |
| Random Forest | Secondary | 0.605 | — |

**❌ H1 Not Supported** — Logistic Regression outperformed Random Forest on both datasets. Simpler, interpretable models generalise better on sparse clinical data.

---

### H2 — Feature Set Comparison
> *"Incorporating utilisation features will significantly improve prediction accuracy."*

| Feature Set | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Demographics Only (race, gender, age) | 0.529 | 0.120 | 0.205 |
| Demographics + Utilisation | **0.629** | **0.189** | **0.243** |

**✅ H2 Supported** — Adding `time_in_hospital`, `number_inpatient`, `number_outpatient`, `number_emergency` improved AUC by **+19%** (0.529 → 0.629).

---

### H3 — Class Imbalance Correction
> *"SMOTE oversampling will improve Recall of readmitted cases."*

| Model | AUC | Recall |
|---|---|---|
| Baseline (no correction) | 0.628 | 0.495 |
| SMOTE | 0.606 | 0.423 |

**⚠️ H3 Mixed** — SMOTE decreased both AUC and Recall on UCI. On Secondary, SMOTE gave a modest Recall improvement while maintaining AUC.

---

## 💻 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM Provider** | Azure OpenAI Service (GPT-4o) |
| **ML Models** | scikit-learn (Logistic Regression, Random Forest) |
| **Data Handling** | Pandas & NumPy |
| **Visualisation** | Matplotlib |
| **Security** | Streamlit Secrets Management |

---

## 📂 Project Structure

```
├── app.py                              # Backend — model training, agents, all pages
├── ui.py                               # UI entry point for Streamlit Cloud
├── cleaned_Diabetic_data__2_.csv       # UCI Diabetes dataset (101,766 rows)
├── cleaned_hospital_readmissions.csv   # Secondary dataset (25,000 rows)
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## 🔧 Local Setup

```bash
git clone https://github.com/temitopeogundare2015-blip/DataraFlow-Internship
cd DataraFlow-Internship
pip install -r requirements.txt
```

Create a `.env` file:
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01
```

```bash
streamlit run app.py
```

---

## ☁️ Streamlit Cloud Deployment

1. Push all 5 files to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → connect your repo
3. Set **Main file path** to `ui.py`
4. Go to **Settings → Secrets** and paste:

```toml
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-key-here"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
AZURE_OPENAI_API_VERSION = "2024-02-01"
```

5. Click **Deploy** ✅

---

## 🖥️ App Pages

| Page | Description |
|---|---|
| 🏠 Home | Research overview, live KPI metrics, agent pipeline diagram |
| 🔬 Patient Triage | Patient form → 3 AI agents → risk gauge + downloadable report |
| 📊 Model Results | Live AUC/F1 charts across H1, H2, H3 for both datasets |
| 📈 Feature Analysis | Feature importance + class distribution comparison |
| 🗄️ Dataset Explorer | Browse both raw datasets with statistics |
| 🤖 Research Assistant | Chat with GPT-4o about your research |

---

## 📚 Key References

1. Kansagara et al. (2011) — Systematic review of readmission risk models, *JAMA*
2. Strack et al. (2014) — UCI Diabetes 130-US Hospitals Dataset, *UCI ML Repository*
3. Artetxe et al. (2018) — Predictive models for hospital readmission, *CMPB*
4. Hai et al. (2023) — Deep Learning vs Traditional Models for Diabetes Readmission, *AMIA*
5. Hu et al. (2022) — Explainable ML for ICU readmission, *Infect. Dis. Ther.*

---

## 👩🏾‍💻 Author

**Temitope Adereni**
- **Role:** Data Science Intern at DataraFlow
- **Programme:** GenAI & Data Science — DF2025-036
- **GitHub:** [@temitopeogundare2015-blip](https://github.com/temitopeogundare2015-blip)
- **Email:** temitopeogundare2015@gmail.com

---

*Developed as part of the DataraFlow Generative AI & Data Science Internship Programme.*
