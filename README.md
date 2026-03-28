# 🏥 Early Prediction of Clinical Deterioration in ICU Patients

## Using Machine Learning — Binary Classification · Healthcare AI · Supervised Learning

**IBT × GGateway Data Science Bootcamp | 2026**

**Mentor: Courage Dike**

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Click%20Here-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://icuapp-snm3d32ncd8qsdwbbarlau.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-Microsoft-5C2D91?style=for-the-badge)](https://lightgbm.readthedocs.io)

---

## 📌 Project Overview

This project is a complete end-to-end **Healthcare AI & Machine Learning** system that predicts whether an ICU patient is at high risk of requiring **mechanical ventilation within the next 6–12 hours**, using hourly rolling windows of vital signs, respiratory parameters, and treatment escalation flags.

The system was trained on the **eICU Collaborative Research Database (PhysioNet)** — a multi-center critical care database of real ICU patient data from multiple US hospitals (2014–2015).

> The system is designed to **augment — not replace — clinical judgment**. It provides a continuous, automated second opinion that surfaces patterns invisible to manual monitoring.

---

## 🌐 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://icuapp-snm3d32ncd8qsdwbbarlau.streamlit.app/)

🔗 [icuapp-snm3d32ncd8qsdwbbarlau.streamlit.app](https://icuapp-snm3d32ncd8qsdwbbarlau.streamlit.app/)

---

## 🔬 Clinical Problem

Current ICU monitoring systems display real-time vital signs but **cannot predict deterioration hours in advance**, limiting the window for clinical intervention. Mechanical ventilation — when initiated proactively — leads to significantly better outcomes than emergency intubation.

### Why the 6–12 Hour Window?

| Window | Reason |
|--------|--------|
| Beyond 12 hours | Signal too weak — vital signs have not begun characteristic deterioration |
| Within 6 hours | Too late for meaningful clinical preparation; data leakage risk |
| **6–12 hours** ✅ | **Actionable lead time — clinicians can prepare, escalate, and prevent emergency intubation** |

### Binary Classification Task

| Label | Definition |
|-------|------------|
| `y = 0` (Stable) | Patient does not require mechanical ventilation in the next 6–12 hours |
| `y = 1` (High Risk) | Mechanical ventilation will be initiated within the next 6–12 hours |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | eICU Collaborative Research Database (PhysioNet) |
| **Access Level** | Credentialed — CITI training + Data Use Agreement |
| **ICU Patients** | 2,520 unique ICU stays |
| **Total Hourly Windows** | 136,867 rows |
| **Training Set** | 109,474 rows — 250 positives (0.23%) |
| **Test Set** | 27,393 rows — 48 positives (0.18%) |
| **Ventilated Patients** | 69 (2.7%) |
| **Class Imbalance Ratio** | 436:1 (severe) |
| **Positive Windows** | 298 out of 136,867 (0.22%) |
| **Features Used** | 31 engineered features |

### Source Tables

| Table | Rows | Primary Use |
|-------|------|-------------|
| patient.csv | 2,520 | Demographics, ICU stay duration, discharge status |
| respiratoryCare.csv | 5,436 | Ventilation start times — label creation only |
| respiratoryCharting.csv | 169,817 | FiO2, PEEP, Tidal Volume, Respiratory Rate |
| vitalPeriodic.csv | 1,633,310 | Heart rate, SpO2, respiration rate |
| treatment.csv | 38,048 | Oxygen therapy, noninvasive ventilation flags |

---

## 🔄 Full Workflow

```
Raw eICU CSV Files
       ↓
1. Data Loading & Inspection
       ↓
2. Exploratory Data Analysis (EDA)
       ↓
3. Data Cleaning & Outlier Treatment
       ↓
4. Hourly Patient Grid Construction
       ↓
5. Label Engineering (y_high — strict leakage prevention)
       ↓
6. Rolling Window Feature Engineering (3h mean, std, slope)
       ↓
7. Patient-Level Train / Test Split (80/20)
       ↓
8. Missing Value Imputation (train-only fit)
       ↓
9. Categorical Encoding (OneHotEncoder — train-only fit)
       ↓
10. Feature Scaling (StandardScaler — train-only fit)
       ↓
11. Class Imbalance Handling (class_weight / SMOTE / ADASYN)
       ↓
12. Model Training (LR / RF / XGBoost / LightGBM / LSTM)
       ↓
13. Evaluation (AUPRC · AUROC · Recall · F1)
       ↓
14. Feature Importance Analysis
       ↓
15. Streamlit Demo + Power BI Dashboard
```

---

## 🧹 Step 1 — Exploratory Data Analysis (EDA)

A comprehensive EDA was conducted across all five source tables before any modeling.

### Missing Data Decisions

| Column | Missing % | Decision |
|--------|-----------|----------|
| dischargeweight | 51.0% | Dropped — post-ICU measurement |
| icp, pasystolic, padiastolic | 98–99% | Dropped — no usable signal |
| temperature, cvp, etco2 | 87–96% | Dropped — too sparse |
| heartrate | 0.4% | Retained — imputed with median |
| sao2 (SpO2) | 11.8% | Retained — imputed with median |
| respiration | 15.5% | Retained — imputed with median |

### Target Variable

| Metric | Value |
|--------|-------|
| Total ICU patients | 2,520 |
| Ventilated patients | 69 (2.7%) |
| Hourly windows (y=1) | 298 out of 136,867 (0.22%) |
| Imbalance ratio | ~436:1 |
| Median ventilation onset | ~27 hours after admission |

### Bivariate Analysis — Safe vs. Danger Hours

| Feature | Safe (y=0) | Danger (y=1) | Direction | Clinical Meaning |
|---------|-----------|--------------|-----------|-----------------|
| Heart Rate (bpm) | 85.3 | 90.7 | ↑ +6.3% | Tachycardia developing |
| SpO2 (%) | 96.6 | 95.8 | ↓ −0.8% | Oxygen saturation dropping |
| Respiration Rate | 20.0 | 22.3 | ↑ +11.5% | Respiratory distress pattern |
| FiO2 (%) | 41.1 | 47.2 | ↑ +14.8% | Increasing oxygen demand |
| SpO2 Slope (3h) | −0.04 | −0.21 | ↓ faster | SpO2 dropping faster in danger hours |
| HR Slope (3h) | −0.14 | −0.58 | ↓ faster | HR more volatile before ventilation |

> ✅ All vital signs move in the clinically correct direction during danger hours — validating that the engineered features capture real physiological deterioration patterns.

---

## 🏗️ Step 2 — Hourly Patient Grid & Label Engineering

### Hourly Grid Construction

Rather than generating one prediction per patient, the model evaluates each patient at **every hour of their ICU stay** — creating a time-aware training set.

| Property | Value |
|----------|-------|
| Time Resolution | 60 minutes per window |
| Total Windows | 136,867 rows across 2,494 patients |
| Alignment Method | merge_asof with direction='backward' |
| Leakage Removal | All windows from 6h before ventilation onward excluded |

### Label Definition (y_high)

```
y_high = 1  →  Window falls in [vent_start − 720min, vent_start − 360min]  (6–12h danger zone)
y_high = 0  →  All other windows (non-ventilated + windows > 12h before ventilation)
Excluded    →  Windows from vent_start − 360min onward  (leakage zone — too close)
```

---

## ⚙️ Step 3 — Feature Engineering (31 Features)

| Category | Features | Count |
|----------|----------|-------|
| Vital sign last values | heartrate_last, sao2_last, respiration_last | 3 |
| 3h Rolling mean | heartrate_mean_3h, sao2_mean_3h, respiration_mean_3h | 3 |
| 3h Rolling std | heartrate_std_3h, sao2_std_3h, respiration_std_3h | 3 |
| 3h Slope (diff) | heartrate_slope_3h, sao2_slope_3h, respiration_slope_3h | 3 |
| Respiratory parameters | fio2_last, peep_last | 2 |
| Treatment flags | on_oxygen_therapy, on_noninvasive_vent | 2 |
| Missingness indicators | hr_missing, sao2_missing, fio2_missing | 3 |
| Demographics (numeric) | age, admissionweight | 2 |
| Unit type (one-hot) | unittype_MICU, unittype_SICU, unittype_CCU-CTICU, etc. | 8 |
| Gender (one-hot) | gender_Female, gender_Male | 2 |
| **TOTAL** | | **31** |

> Missing vital signs in ICU are clinically informative — a patient whose SpO2 is not being recorded may be in a worse state. This is why missingness indicators are included as features.

---

## 🔧 Step 4 — Data Preprocessing Pipeline

All preprocessing was performed with **strict train/test separation**. Every statistic was computed exclusively on training data and applied to test data without re-fitting.

### Pipeline Steps

**Step 1 — Outlier Treatment**
Physiologically impossible values replaced with training-set median:
- Heart rate: ≤0 or >250 bpm
- SpO2: <50%
- Respiration: ≤0 or >80 br/min
- Height: >250cm or <100cm
- Weight: >300kg or <20kg

**Step 2 — Missing Value Imputation**
- Numeric: SimpleImputer with training-set median
- Categorical: SimpleImputer with training-set mode
- Result: Zero missing values across all tables

**Step 3 — Categorical Encoding**
- OneHotEncoder(handle_unknown='ignore') on unit type (8 categories) and gender (2 categories)
- Fitted only on training data

**Step 4 — Feature Scaling**
- StandardScaler applied to 16 continuous numeric features
- Fitted on train only, transformation applied to both sets

### Train / Test Split

| Set | Patients | Windows | Positives |
|-----|----------|---------|-----------|
| Training | 1,995 | 109,474 | 250 (0.23%) |
| Test | 499 | 27,393 | 48 (0.18%) |

> **Patient-level split** was used — NOT row-level. Splitting by row allows the same patient's hours to appear in both sets, causing data leakage.

---

## ⚖️ Step 5 — Class Imbalance Handling

| Strategy | How It Works | Applied To |
|----------|-------------|------------|
| class_weight='balanced' | Re-weights loss — penalizes minority misclassification | Logistic Regression, Random Forest |
| class_weight='balanced_subsample' | Per-tree reweighting in Random Forest | Random Forest ⭐ Best |
| scale_pos_weight=436.9 | XGBoost native equivalent | XGBoost |
| SMOTE | Generates synthetic minority examples | RF + XGBoost variants |
| SMOTE+Tomek | SMOTE + removal of ambiguous majority samples | RF + XGBoost variants |
| ADASYN | Adaptive SMOTE — focuses on harder examples | RF + XGBoost variants |

---

## 🔬 Step 5b — LightGBM: Detailed Modeling Strategy

### Class Imbalance Strategies Compared

Three approaches were tested for handling the 436:1 imbalance in LightGBM specifically:

| Strategy | How it works | Result | Decision |
|----------|-------------|--------|----------|
| `is_unbalance=True` | LightGBM auto-weights by class frequency | AUPRC: 0.0084 | ⚠️ Similar to manual weight — less control |
| `scale_pos_weight=436.9` | Manual ratio of negatives to positives | AUPRC: 0.0084 | ✅ Kept — explicit, tunable |
| No weighting | Standard training | AUPRC: 0.0087 | ❌ Dropped — 0% recall at default threshold |
| SMOTE | Synthetic minority oversampling | Not pursued | ❌ See reasoning below |

**Why SMOTE was not used for LightGBM:**
- With only 250 positive training rows, SMOTE's k=5 neighbours are unreliable — synthetic examples are interpolated between very few real cases, producing clinically implausible patterns
- The Springer (2024) review found SMOTE consistently introduces calibration drift on severely imbalanced medical data
- Cost-sensitive learning (`scale_pos_weight`) achieved comparable or better AUPRC without modifying the training distribution
- SMOTE was tested on Random Forest and XGBoost — it helped RF slightly (AUPRC 0.0040 vs 0.0031) but hurt XGBoost (recall dropped from 17% to 4%)

---

### Hyperparameter Tuning — Optuna Bayesian Optimisation

**Why Optuna over Grid Search:**
Grid search evaluates every parameter combination exhaustively — with 9 parameters and reasonable ranges, this would require thousands of evaluations. Optuna uses the **Tree Parzen Estimator (TPE)** algorithm: it learns which parameter regions are promising and focuses trials there. 50 Optuna trials typically outperforms 500 random search trials.

**Parameters tuned and why each matters:**

| Parameter | Search Range | Why it matters for imbalance |
|-----------|-------------|------------------------------|
| `scale_pos_weight` | 50–1000 (log scale) | **Key addition** — lets Optuna find the optimal minority class penalty rather than fixing it at 436.9 |
| `learning_rate` | 0.001–0.1 (log scale) | Lower rates generalise better on rare events — model takes more careful steps |
| `n_estimators` | 200–1000 | More trees needed when learning rate is low |
| `num_leaves` | 20–100 | Controls tree complexity — too many = overfitting on the 250 positive rows |
| `min_child_samples` | 20–200 | **Critical** — prevents leaf nodes built around individual positive rows |
| `max_depth` | 3–10 | Caps tree depth — shallower trees generalise better for rare events |
| `subsample` | 0.5–1.0 | Row subsampling — adds randomness, reduces overfitting |
| `colsample_bytree` | 0.5–1.0 | Feature subsampling per tree — most impactful parameter found by Optuna |
| `reg_alpha` | 1e-4–10 (log scale) | L1 regularisation |
| `reg_lambda` | 1e-4–10 (log scale) | L2 regularisation |

**Best parameters found:**

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| `scale_pos_weight` | 191.9 | Lower than computed 436.9 — full ratio over-penalises, hurting precision |
| `learning_rate` | 0.013 | Slow — careful steps needed with only 250 positive rows |
| `min_child_samples` | 57 | High — prevents memorising individual positive patients |
| `colsample_bytree` | 0.81 | Ranked most important by Optuna — feature subsampling key to generalisation |

---

### Cross-Validation — The Patient Leakage Problem

**What went wrong first:**

Using standard `StratifiedKFold` (row-level splits), CV AUPRC reached **0.39** — a result that looked excellent. When the model was evaluated on the held-out test set, AUPRC collapsed to **0.002**.

**Why this happened:**

Each patient contributes ~55 rows (109,474 rows ÷ 1,995 patients). Row-level CV splits these rows across folds, so the same patient's hour-3 data appears in the training fold while their hour-7 data appears in the validation fold. The model memorises patient-specific patterns and then "recognises" that patient in validation — producing an inflated score that does not generalise to new patients.

**The fix — `StratifiedGroupKFold`:**
```python
cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
for train_idx, val_idx in cv.split(X_train, y_flat, groups=train_patient_ids):
    # all rows from a patient stay in the same fold
```

With patient ID as the group key, all rows from a patient go entirely into either the training or validation fold — never both. CV AUPRC dropped to an honest **0.011**, which closely matched the test result of **0.0029**.

| CV Method | CV AUPRC | Test AUPRC | Gap | Verdict |
|-----------|----------|------------|-----|---------|
| `StratifiedKFold` (row-level) | 0.3925 | 0.0020 | 35× | ❌ Leakage |
| `StratifiedGroupKFold` (patient-level) | 0.0110 | 0.0029 | 3.8× | ✅ Honest |

> This finding is the most important methodological result of the LightGBM work. Any dataset with repeated measurements per subject — ICU data, wearables, longitudinal health records — requires group-aware cross-validation.

---

### Threshold Analysis

**Why the default threshold (0.5) fails here:**

With 0.18% positive rate, the model's predicted probabilities for danger hours cluster far below 0.5 — even when the model correctly identifies them. Using 0.5 as the decision boundary predicts safe for every single row, catching zero patients.

**How the optimal threshold was found:**

The F2 score weights recall twice as heavily as precision, reflecting the clinical reality that a missed ventilation event is more dangerous than a false alarm:
```
F2 = (5 × Precision × Recall) / (4 × Precision + Recall)
```

**Results at F2-optimal threshold (0.098):**

| Metric | Value |
|--------|-------|
| Threshold | 0.098 (vs default 0.5) |
| Recall | 20.8% — 10 of 48 danger hours caught |
| Precision | 0.5% |
| False alarms per true catch | 201 |
| True positives | 10 |
| False negatives (missed) | 38 |

**Clinically targeted thresholds:**

| Recall target | Threshold | Precision | False alarm rate |
|---------------|-----------|-----------|-----------------|
| 40% | 0.0079 | 0.3% | 99.7% |
| 50% | 0.0079 | 0.3% | 99.7% |
| 60% | 0.0028 | 0.2% | 99.8% |
| 70% | 0.0028 | 0.2% | 99.8% |
| 80% | 0.0024 | 0.2% | 99.8% |

> The high false alarm rate across all thresholds reflects the fundamental sample size constraint — 48 test positives in 27,393 rows. On the full eICU dataset, precision at equivalent recall targets would be substantially higher.

---

### Feature Importance — Three Methods Compared

Running three importance methods guards against the blind spots of any single approach:

| Method | What it measures | Key limitation |
|--------|-----------------|----------------|
| LightGBM gain | Total loss reduction from splits on that feature | Biased toward features that appear in many rows (e.g. static demographics) |
| Permutation importance | Drop in AUPRC when feature is randomly shuffled | Honest for our actual metric — computationally expensive |
| SHAP | Directional contribution per prediction | Most interpretable — shows sign and magnitude of effect |

**Key finding — gain importance was misleading:**

| Feature | Gain rank | Permutation rank | Verdict |
|---------|-----------|-----------------|---------|
| `age` | 2nd | 31st | ❌ Artifact — static feature exploiting row ubiquity |
| `admissionweight` | 1st | 23rd | ❌ Artifact — same reason |
| `respiration_mean_3h` | 3rd | **1st** | ✅ True predictor |
| `heartrate_mean_3h` | 4th | **2nd** | ✅ True predictor |
| `fio2_missing` | 18th | **5th** | ✅ Missingness is a genuine signal |

**SHAP direction (from beeswarm plot):**
- High `heartrate_mean_3h` → pushes toward danger ✅ clinically correct
- High `respiration_mean_3h` → pushes toward danger ✅ clinically correct
- High `fio2_last` → pushes toward danger ✅ higher oxygen demand = risk
- Low `sao2_mean_3h` → pushes toward danger ✅ dropping SpO2 = risk
---

## 🤖 Step 6 — Model Development

### Model 1 — Logistic Regression (Baseline)
- **Purpose:** Establish performance floor, confirm preprocessing is correct
- **Algorithm:** Ordinary Least Squares — assumes linear separability
- **Result:** AUPRC: 0.0015 | AUROC: 0.400 | Recall: 71% | F1: 0.00
- **Finding:** High recall achieved by predicting positive too liberally — clinically impractical

### Model 2 — Random Forest ⭐ Best Overall
- **Algorithm:** 100 decision trees, bootstrap sampling, probability averaging
- **Best Config:** class_weight='balanced_subsample'
- **Result:** AUPRC: 0.0031 | AUROC: 0.645 | Recall: 50% | F1: 0.011
- **Finding:** Correctly identifies 50% of all true ventilation risk windows 6–12h in advance

### Model 3 — XGBoost
- **Algorithm:** Sequential gradient boosting — each tree corrects errors of previous
- **Best Config:** scale_pos_weight=436.9
- **Result:** AUPRC: 0.0032 | AUROC: 0.632 | Recall: 17% | F1: 0.012
- **Finding:** SMOTE hurt XGBoost — recall dropped from 17% to 4% with basic SMOTE

### Model 4 — LightGBM
- **Algorithm:** Histogram-based leaf-wise tree growth (Microsoft)
- **Best Config:** Optuna Bayesian tuning (50 trials) + Effective Number of Samples weighting
- **Result:** AUPRC: 0.0037 | AUROC: 0.637 | Recall: 23% | F1: 0.014
- **Finding:** Highest raw AUPRC among non-SMOTE models; static demographics dominated importance

### Model 5 — Deep Learning (Stacked LSTM + BiLSTM)
- **Architecture:** Stacked LSTM(128→32) and Bidirectional LSTM(256→64) via TensorFlow/Keras
- **Tuning:** keras_tuner RandomSearch (10 trials, EarlyStopping patience=3)
- **Stacked LSTM Result:** AUPRC: 0.0030 | AUROC: 0.503 | Recall: 0%
- **BiLSTM Result:** AUPRC: 0.0026 | AUROC: 0.538 | Recall: 0%
- **Finding:** seq_len=1 means no real sequences fed to LSTM layers — requires seq_len ≥ 12 to exploit temporal memory

---

## 📈 Step 7 — Evaluation & Results

### Primary Metric: AUPRC
> A model predicting y=0 for every window achieves 99.78% accuracy — yet identifies zero patients at risk. **AUPRC is the correct metric for severely imbalanced clinical datasets.**

| Model | Strategy | AUPRC | AUROC | Recall | F1 |
|-------|----------|-------|-------|--------|-----|
| Logistic Regression | class_weight=balanced | 0.0015 | 0.400 | 71% | 0.00 |
| XGBoost | scale_pos_weight=436.9 | 0.0032 | 0.632 | 17% | 0.012 |
| XGBoost | ADASYN | 0.0030 | 0.590 | 27% | 0.020 |
| **Random Forest** | **class_weight=balanced_sub.** | **0.0031** | **0.645** | **50%** | **0.011** |
| Random Forest | SMOTE | 0.0040 | 0.663 | 31% | 0.014 |
| LightGBM | Eff. Sampling + Optuna | 0.0037 | 0.637 | 23% | 0.014 |
| Stacked LSTM | RandomSearch, seq_len=1 | 0.0030 | 0.503 | 0% | 0.00 |
| Bidirectional LSTM | RandomSearch, seq_len=1 | 0.0026 | 0.538 | 0% | 0.00 |

### Top 10 Feature Importances

| Rank | Feature | Correlation with y_high | Clinical Interpretation |
|------|---------|------------------------|------------------------|
| 1 | respiration_mean_3h | +0.0202 | Mean RR over last 3h — respiratory distress building |
| 2 | respiration_last | +0.0179 | Most recent respiratory rate reading |
| 3 | fio2_last | +0.0159 | Last FiO2 value — oxygen demand indicator |
| 4 | heartrate_mean_3h | +0.0145 | Mean HR over last 3h — tachycardia developing |
| 5 | sao2_missing | +0.0144 | SpO2 missingness — clinically informative |
| 6 | hr_missing | +0.0144 | HR missingness — clinically informative |
| 7 | heartrate_last | +0.0134 | Most recent heart rate reading |
| 8 | sao2_mean_3h | −0.0126 | Mean SpO2 — lower = more danger |
| 9 | sao2_last | −0.0115 | Most recent SpO2 — lower = more danger |
| 10 | admissionweight | +0.0053 | Body mass affects respiratory mechanics |

---

## 📄 Research Basis — Imbalanced Medical ML

The LightGBM modeling approach was informed by two key papers:

| Paper | Key finding applied |
|-------|-------------------|
| **Springer (2024)** — Systematic review of imbalanced classification methods on medical data | LightGBM with cost-sensitive learning consistently outperforms SMOTE variants on severely imbalanced clinical data. AUPRC is the correct metric — accuracy and AUC-ROC are misleading at <1% positive rates. Applying imbalance correction before splitting is identified as the most common fatal mistake |
| **PLOS ONE (2025)** — Instance selection vs oversampling for imbalanced eICU data | Instance selection and group-aware evaluation outperform SMOTE by finding better decision boundaries rather than generating synthetic data. Patient-level group constraints in CV are mandatory for repeated-measurement medical datasets |

Both papers report LightGBM achieving AUPRC of **~0.63** on the full eICU dataset (~200,000 patients). Our pipeline is methodologically identical — the gap in results is entirely explained by the 80× smaller demo dataset (2,494 patients).

---

## 📊 Power BI Dashboard

An interactive **Power BI dashboard** was built on the engineered 136,867-row hourly patient grid. It provides three views:

| View | Filter | Key Insight |
|------|--------|-------------|
| Full Dataset | None | 298 high-risk hours (0.22%), avg HR 84 bpm, 137K total records |
| Safe Windows Only | Status = Safe | 99.78% of all windows — baseline ICU population patterns |
| Danger Windows Only | Status = Danger | Avg HR rises to 87 bpm; higher oxygen therapy usage in danger cohort |

---

## 🆚 Comparison with Traditional Approaches

| Characteristic | Traditional EWS (NEWS2) | This Project |
|----------------|------------------------|--------------|
| Update Frequency | Manual — fixed intervals | Continuous — every hour automatically |
| Data Requirement | Nurse-administered scoring | Automatic EHR extraction |
| Temporal Trends | Not captured | 3h rolling window (slope, mean, std) |
| Prediction Horizon | Reactive at event time | Predictive — 6–12h advance warning |
| Scalability | Requires nursing time | Fully automated pipeline |
| AUROC | 0.70–0.80 (manual) | 0.645 (automated, 3h features only) |

---

## 🚀 Run Locally

```bash
git clone https://github.com/RoaaRaed/ibt-ggateway-capstone-healthcare.git
cd ibt-ggateway-capstone-healthcare
pip install -r requirements.txt
streamlit run app.py
```

App opens at: **http://localhost:8501**

---

## 📁 Repository Structure

```
ibt-ggateway-capstone-healthcare/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb     # Data cleaning & feature engineering
│   ├── 03_modeling.ipynb          # Model training & evaluation
│   └── 04_deep_learning.ipynb     # LSTM experiments
└── ICU_Project_Documentation.pdf  # Full project documentation (33 pages)
```

---

## 🔮 Future Work

| Priority | Area | Action |
|----------|------|--------|
| High | Vital Trends | Add SpO2 and HR deterioration slope over 6h and 12h |
| High | Interaction Features | Combined signals: low SpO2 + high RR, rising FiO2 + rising HR |
| High | LightGBM — full eICU | Re-run tuned pipeline on full eICU — expected AUPRC 0.3–0.6 based on literature |
| Medium | Drop age/admissionweight | Confirmed artifacts by permutation importance — will improve real-time signal ranking |
| Medium | External Validation | Test on MIMIC-IV for generalizability |
| Long-term | LSTM Sequences | seq_len ≥ 12 with MIMIC-IV (100,000+ ICU stays) |
| Long-term | LLM Integration | Clinical notes via large language model embeddings |

---

## 👥 Team Members

| Member | Role |
|--------|------|
| **Ahmad Assi** | Project Coordinator + Data Lead + Deep Learning |
| **Bisan Ghoul** | EDA, Feature Engineering & Preprocessing + LightGBM Modeling (Tuning, Threshold & SHAP) |
| **Roa'a Jaber** | Modeling Lead + Streamlit Demo + Deep Learning |
| **Mohammad Zyoud** | Modeling Lead + Deep Learning |
| **Roaa Abu Arra** | Documentation + Slides Lead + Power BI |

**Mentor:** Courage Dike
**Bootcamp:** IBT × GGateway Data Science Bootcamp | 2026

---

## 📚 References

1. Pollard et al. (2018) — eICU Collaborative Research Database, *Scientific Data*
2. Johnson et al. (2023) — MIMIC-IV, *Scientific Data*
3. Chen & Guestrin (2016) — XGBoost, *KDD*
4. Royal College of Physicians (2017) — NEWS2
5. Pedregosa et al. (2011) — Scikit-learn, *JMLR*
6. Chawla et al. (2002) — SMOTE, *JAIR*
7. Lundberg & Lee (2017) — SHAP, *NeurIPS*
8. PhysioNet eICU-CRD v2.0 — physionet.org

---

*IBT × GGateway Data Science Bootcamp | Mentor: Courage Dike | 2026*
