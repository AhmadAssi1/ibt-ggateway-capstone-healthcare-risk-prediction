# 🏥 Early Prediction of Clinical Deterioration in ICU Patients

## Using Machine Learning — Binary Classification · Healthcare AI

**IBT × GGateway Data Science Bootcamp | 2026**

**Mentor: Courage Dike**

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Click%20Here-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://icuapp-snm3d32ncd8qsdwbbarlau.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![ML](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

## 📌 Project Overview

This project is a complete end-to-end **Healthcare AI & Machine Learning** system that predicts whether an ICU patient is at high risk of requiring **mechanical ventilation within the next 6–12 hours**, using hourly rolling windows of vital signs, respiratory parameters, and treatment escalation flags.

The system was trained on the **eICU Collaborative Research Database (PhysioNet)** — a multi-center critical care database of real ICU patient data.

---

## 🌐 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://icuapp-snm3d32ncd8qsdwbbarlau.streamlit.app/)

🔗 [icuapp-snm3d32ncd8qsdwbbarlau.streamlit.app](https://icuapp-snm3d32ncd8qsdwbbarlau.streamlit.app/)

---

## 🔬 Clinical Problem

Current ICU monitoring systems display real-time vital signs but **cannot predict deterioration hours in advance**, limiting the window for clinical intervention.

> **Research Question:** Can we predict whether an ICU patient is at high risk of requiring mechanical ventilation within the next 6–12 hours using historical vital signs and clinical data from the eICU dataset?

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | eICU Collaborative Research Database (PhysioNet) |
| **ICU Patients** | 2,520 unique ICU stays |
| **Total Hourly Windows** | 136,867 rows |
| **Ventilated Patients** | 69 (2.7%) |
| **Class Imbalance** | 436:1 (severe) |
| **Positive Windows** | 298 out of 136,867 (0.22%) |

---

## 🤖 Machine Learning Models

| Model | Strategy | AUROC | Recall | Notes |
|-------|----------|-------|--------|-------|
| Logistic Regression | class_weight=balanced | 0.400 | 71% | Baseline |
| Random Forest | class_weight=balanced_subsample | **0.645** | **50%** | ⭐ Best Overall |
| Random Forest | SMOTE | 0.663 | 31% | Higher AUPRC, lower recall |
| XGBoost | scale_pos_weight=436.9 | 0.632 | 17% | Native imbalance handling |
| LightGBM | Effective Sampling + Optuna | 0.637 | 23% | Highest AUPRC non-SMOTE |
| Stacked LSTM | RandomSearch, seq_len=1 | 0.503 | 0% | Needs true sequences |
| Bidirectional LSTM | RandomSearch, seq_len=1 | 0.538 | 0% | Needs true sequences |

**Best Model: Random Forest + class_weight='balanced_subsample'**
- AUPRC: 0.0031 | AUROC: 0.645 | Recall: 50% | F1: 0.011
- Correctly identifies **50% of all true ventilation risk windows — 6–12 hours in advance**

---

## ⚙️ Feature Engineering

The core innovation is converting each ICU stay into an **hourly evaluation grid** with 31 engineered features:

| Category | Features | Count |
|----------|----------|-------|
| Vital sign last values | heartrate_last, sao2_last, respiration_last | 3 |
| 3h Rolling mean | heartrate_mean_3h, sao2_mean_3h, respiration_mean_3h | 3 |
| 3h Rolling std | heartrate_std_3h, sao2_std_3h, respiration_std_3h | 3 |
| 3h Slope | heartrate_slope_3h, sao2_slope_3h, respiration_slope_3h | 3 |
| Respiratory parameters | fio2_last, peep_last | 2 |
| Treatment flags | on_oxygen_therapy, on_noninvasive_vent | 2 |
| Missingness indicators | hr_missing, sao2_missing, fio2_missing | 3 |
| Demographics | age, admissionweight, unittype (OHE), gender (OHE) | 12 |
| **Total** | | **31** |

---

## 🛡️ Key Methodology

- **Patient-level train/test split** — prevents data leakage from same-patient hours
- **merge_asof with direction='backward'** — carries forward last known value to each window
- **6–12 hour prediction window** — actionable lead time before ventilation onset
- **AUPRC as primary metric** — correct for severe class imbalance (not accuracy)
- **Strict leakage prevention** — post-ventilation windows excluded from training

---

## 🚀 Run Locally

```bash
git clone https://github.com/RoaaRaed/ibt-ggateway-capstone-healthcare.git
cd ibt-ggateway-capstone-healthcare
pip install -r requirements.txt
streamlit run app.py
```

---

## 👥 Team Members

| Member | Role |
|--------|------|
| **Ahmad Assi** | Project Coordinator + Data Lead + Deep Learning |
| **Bisan Ghoul** | EDA & Visualization Lead |
| **Roa'a Jaber** | Modeling Lead + Streamlit Demo + Deep Learning |
| **Mohammad Zyoud** | Modeling Lead + Deep Learning |
| **Roaa Abu Arra** | Documentation + Slides Lead + Power BI |

**Mentor:** Courage Dike | **Bootcamp:** IBT × GGateway Data Science Bootcamp

---

## 📈 Key Results

- **Best AUROC:** 0.645 (Random Forest)
- **Best Recall:** 50% — detects 1 in 2 high-risk patients 6–12 hours before ventilation
- **Top Feature:** respiration_mean_3h — respiratory distress pattern
- **Primary Metric:** AUPRC (appropriate for 436:1 class imbalance)

---

## 📁 Repository Structure

```
ibt-ggateway-capstone-healthcare/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── notebooks/                      # Jupyter notebooks (EDA, modeling)
└── ICU_Project_Documentation.pdf  # Full project documentation
```

---

## 📚 References

- Pollard et al. (2018) — eICU Collaborative Research Database, Scientific Data
- Chen & Guestrin (2016) — XGBoost, KDD
- Chawla et al. (2002) — SMOTE, JAIR
- PhysioNet eICU-CRD v2.0 — physionet.org

---

*IBT × GGateway Data Science Bootcamp | Mentor: Courage Dike | 2026*
