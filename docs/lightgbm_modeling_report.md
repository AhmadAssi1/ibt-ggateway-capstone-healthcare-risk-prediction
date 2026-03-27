# LightGBM — Hyperparameter Tuning, Threshold Analysis & SHAP
### ICU Mechanical Ventilation Risk Prediction | eICU Collaborative Research Database Demo v2.0.1

---

## Overview

This notebook builds and evaluates a LightGBM classifier to predict whether an ICU patient
will require mechanical ventilation in the **next 6–12 hours**, using hourly vital sign data
from the eICU Collaborative Research Database.

The core challenge is **severe class imbalance** — only 0.22% of hourly rows are positive
(danger hours). This notebook documents a rigorous approach to handling that imbalance,
including a critical discovery of patient leakage in cross-validation that inflated
performance estimates by 35×.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | eICU Collaborative Research Database Demo v2.0.1 |
| Patients | 2,494 |
| Total rows | 136,867 hourly observations |
| Positive rate | 0.22% (danger hours) |
| Train/test split | 80/20 by **patient** (not row) |
| Train positives | 250 rows across 1,995 patients |
| Test positives | 48 rows across 499 patients |

> ⚠️ This notebook uses the **demo** version of eICU (~2,500 patients).
> The full dataset contains ~200,000 patients. Results are limited by sample size
> and should be interpreted as a methodological proof of concept.

---

## Features

All features were engineered in the preprocessing notebook from 5 raw eICU tables.

| Feature | Type | Description |
|---------|------|-------------|
| `heartrate_last` | Vital | Most recent heart rate reading |
| `sao2_last` | Vital | Most recent SpO2 reading |
| `respiration_last` | Vital | Most recent respiratory rate |
| `fio2_last` | Respiratory | Most recent fraction of inspired oxygen |
| `peep_last` | Respiratory | Most recent PEEP value |
| `heartrate_mean_3h` | Trend | 3-hour rolling mean of heart rate |
| `heartrate_std_3h` | Trend | 3-hour rolling std of heart rate |
| `heartrate_slope_3h` | Trend | 3-hour directional change in heart rate |
| `sao2_mean/std/slope_3h` | Trend | Same for SpO2 |
| `respiration_mean/std/slope_3h` | Trend | Same for respiratory rate |
| `on_oxygen_therapy` | Treatment | Binary flag |
| `on_noninvasive_vent` | Treatment | Binary flag — escalation signal |
| `hr_missing` / `sao2_missing` / `fio2_missing` | Missingness | Missing vitals are clinically informative |
| `age` / `admissionweight` | Demographics | Static patient features |
| `unittype` | Demographics | ICU type (one-hot encoded) |
| `gender` | Demographics | One-hot encoded |

---

## Notebook Structure

### Section 1 — Data Loading & Baseline Verification
- Row-level and patient-level imbalance check
- Patient leakage verification (zero overlap confirmed)
- SMOTE leakage heuristic check
- `scale_pos_weight` computation (436.9)

### Section 2 — LightGBM Baseline with Cost-Sensitive Learning
Three configurations compared:
- Config A: `is_unbalance=True`
- Config B: `scale_pos_weight=436.9`
- Config C: No weighting

Best baseline AUPRC: **0.0087**

### Section 3 — Optuna Hyperparameter Tuning
- 50 Bayesian optimisation trials via Optuna TPE sampler
- `scale_pos_weight` included in the search space
- **Critical finding:** `StratifiedKFold` (row-level) inflated CV AUPRC to **0.39** due to patient leakage
- Fixed with `StratifiedGroupKFold` — honest CV AUPRC: **0.011**
- Final test AUPRC after retraining: **0.0029**

### Section 4 — Threshold Analysis
- Default threshold (0.5) catches **zero** patients
- F2-optimal threshold found: **0.098**
- At optimal threshold: 10/48 danger hours caught (20.8% recall), 201 false alarms per true catch
- Clinically targeted threshold table: precision at 40–80% recall targets

### Section 5 — Feature Importance & SHAP
Three methods compared:
- LightGBM native gain importance
- Permutation importance on AUPRC (most honest)
- SHAP values (directional, per-prediction)

Key findings:
- `respiration_mean_3h` and `heartrate_mean_3h` are the true top predictors (permutation rank 1 & 2)
- `age` and `admissionweight` rank 1st/2nd by gain but 31st/23rd by permutation — confirmed artifacts
- `fio2_missing` ranks 5th by permutation — missingness is a genuine clinical signal

### Section 6 — Conclusions & Future Work
- Summary of all findings
- Dataset size analysis vs full eICU benchmarks
- Prioritised future work roadmap

---

## Key Findings

### 1. Patient Leakage in CV is Catastrophic
Using `StratifiedKFold` (row-level splits) on patient time-series data
inflated AUPRC from **0.011 → 0.39** — a **35× overestimate**.
`StratifiedGroupKFold` with patient ID as the group key is mandatory
for any dataset with repeated measurements per subject.

### 2. Threshold Selection is Non-Negotiable
With 0.18% positive rate, the default threshold of 0.5 predicts safe
for every row. The optimal threshold was **0.098** — found via F2
optimisation which weights recall twice as heavily as precision.

### 3. Gain Importance Can Mislead
Static features (`age`, `admissionweight`) dominated gain importance
rankings but showed near-zero permutation importance — confirming they
add no real predictive value. Always cross-validate importance with
permutation or SHAP.

### 4. Real-Time Trends Beat Snapshots
3-hour rolling means of respiratory rate and heart rate were the top
true predictors — consistent with clinical practice where deterioration
is detected through trends, not single readings.

---

## Results Summary

| Model | CV AUPRC | Test AUPRC | AUC-ROC |
|-------|----------|------------|---------|
| Section 2 baseline | — | 0.0087 | 0.61 |
| Optuna v1 (StratifiedKFold — leaked) | 0.3925 | 0.0020 | 0.48 |
| **Optuna v2 (StratifiedGroupKFold)** | **0.0110** | **0.0029** | **0.63** |

> Low AUPRC is a consequence of the demo dataset size (250 positive training rows).
> The published literature reports LightGBM achieving AUPRC of ~0.63 on the full
> eICU dataset for equivalent tasks (Springer 2024 review).

---

## Requirements

```
lightgbm
optuna
shap
scikit-learn
pandas
numpy
matplotlib
```

---

## Related Notebooks

| Notebook | Description |
|----------|-------------|
| `1_0_bg_eda.py` | EDA, feature engineering, preprocessing & patient-level train/test split |
| This notebook | LightGBM modeling — tuning, threshold analysis & SHAP |

---

## Author

**Bisan Ghoul** — EDA, Feature Engineering & Preprocessing Lead + LightGBM Modeling (Tuning, Threshold & SHAP)  
eICU ICU Ventilation Risk Prediction Project
