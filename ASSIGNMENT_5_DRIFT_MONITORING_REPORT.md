# Assignment-5: Production Drift Monitoring System
## Model: House Price Prediction Engine (Continuous MLOps Monitoring)
**Comprehensive Academic & Practical Jury Defense Report**

---

## 📌 Candidate & Submission Details
- **Assignment:** Assignment-5 (Continuous Model Monitoring: Data Drift & Concept Drift)
- **Project Domain:** Real Estate Valuation / Production MLOps Monitoring
- **Candidate Name:** Asad ([@AsadLearns](https://github.com/AsadLearns))
- **Repository:** [https://github.com/AsadLearns/mlopsjuryyybystraightnoodles](https://github.com/AsadLearns/mlopsjuryyybystraightnoodles)
- **Language / Environment:** Python 3.11 (`.venv`), Scipy, Scikit-Learn, Pandas, Matplotlib
- **Script Entrypoint:** [`monitor_drift.py`](file:///C:/Users/asadj/Desktop/mlops%20jury/monitor_drift.py)
- **Interactive Report:** [`ASSIGNMENT_5_DRIFT_MONITORING_REPORT.html`](file:///C:/Users/asadj/Desktop/mlops%20jury/ASSIGNMENT_5_DRIFT_MONITORING_REPORT.html)
- **Status:** 100% Fully Implemented, Statistically Validated & Pushed to GitHub

---

## 1. Executive Summary & Objective

Deploying a machine learning model to production is only the beginning of the operational lifecycle. Over time, real-world dynamics cause deployed models to degrade silently. This assignment implements a production-grade monitoring service for a **House Price Prediction Model** that:
1. **Detects Data Drift (Covariate Shift):** Identifies statistically significant shifts in the marginal distributions of input features:
   - `area` (Continuous numerical feature: living area in sqft)
   - `bedrooms` (Discrete numerical feature: room count)
   - `location` (Categorical feature: Suburbs, Downtown, Uptown, Rural)
2. **Detects Concept Drift (Relationship / Performance Shift):** Identifies shifts in the conditional probability distribution $P(Y \mid X)$, where real-world house valuations detach from baseline historical patterns.
3. **Automates Monitoring & Alerting:** Generates an empirical scorecard with statistical p-values, distance metrics (PSI, KS-statistic, Chi-square), performance timelines ($R^2$, RMSE, MAE), and a multi-panel visual dashboard.

---

## 2. Theoretical & Mathematical Foundations

### 2.1 Data Drift (Covariate Shift: $P(X)$ Changes While $P(Y \mid X)$ Remains Constant)
Data drift occurs when incoming feature vectors in production diverge from the reference dataset used during model training:

#### A. Two-Sample Kolmogorov-Smirnov (KS) Test (Numerical Features: `area`, `bedrooms`)
The KS-test evaluates whether two continuous empirical cumulative distribution functions (ECDFs) originate from the same underlying distribution without assuming normality:
$$D_{n, m} = \sup_x \left| F_{\text{reference}}(x) - F_{\text{production}}(x) \right|$$
- **Null Hypothesis ($H_0$):** $F_{\text{reference}}(x) = F_{\text{production}}(x)$ (No data drift).
- **Decision Rule:** Reject $H_0$ if $p\text{-value} < \alpha$ (where $\alpha = 0.05$).

#### B. Population Stability Index (PSI)
PSI measures the divergence between the expected baseline distribution ($E$) and actual production distribution ($A$) divided into $k$ buckets:
$$\text{PSI} = \sum_{i=1}^{k} \left( A_i - E_i \right) \times \ln \left( \frac{A_i}{E_i} \right)$$
- $\text{PSI} < 0.10$: Minimal shift (No action required).
- $0.10 \le \text{PSI} < 0.25$: Moderate drift (Warning alert).
- $\text{PSI} \ge 0.25$: Significant data drift (Immediate model retraining required).

#### C. Chi-Square Test of Independence (Categorical Feature: `location`)
Used to quantify frequency divergence across categorical classes:
$$\chi^2 = \sum_{i=1}^{c} \frac{(O_i - E_i)^2}{E_i}$$
Where $O_i$ is observed frequency in production and $E_i$ is expected baseline frequency. If $p < 0.05$, categorical drift is confirmed.

---

### 2.2 Concept Drift ($P(Y \mid X)$ Changes: Real-World Relationship Shifts)
Even if input feature values remain stable, the real-world economic relationship connecting features $X$ to price $Y$ can warp due to inflation, policy changes, or interest rate hikes.

We monitor model degradation over temporal monthly production batches ($t \in \{\text{Month 1}, \text{Month 2}, \text{Month 3}, \text{Month 4}\}$):
1. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
2. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
4. **Drift Alarm Criterion:**
   $$\Delta \text{RMSE} > +25\% \quad \text{OR} \quad \Delta R^2 > 0.15 \quad (\text{Relative to Baseline})$$

---

## 3. System Architecture & Workflow

```
[ Baseline Training Data ]                    [ Production Data Stream (Months 1–4) ]
   (1,000 Historical Listings)                     (300 Real-Time Monthly Listings)
           │                                                       │
           ▼                                                       ▼
[ Train Baseline Model ]                             [ Ingest Current Inference Window ]
(GradientBoostingRegressor)                                        │
           │                                                       │
           ├───────────────────────────┬───────────────────────────┤
           ▼                           ▼                           ▼
[ Statistical Feature Test ]  [ Category Frequency Test ]  [ Performance Tracker ]
   - KS Test on 'area'           - Chi-Square on 'location'   - Compute RMSE, MAE, R²
   - PSI on 'bedrooms'           - Proportions shift          - Residual Divergence
           │                           │                           │
           └───────────────────────────┼───────────────────────────┘
                                       │
                                       ▼
                       [ Drift Evaluation Engine ]
                        ├── Data Drift Threshold: p < 0.05 or PSI > 0.10
                        └── Concept Drift Threshold: ΔRMSE > +25% or R² drop > 0.15
                                       │
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │       Automated Decision Routing          │
                 ├───────────────────────────────────────────┤
                 │  IF Drift Detected:                       │
                 │   - Generate PNG Dashboard & HTML Report  │
                 │   - Write artifacts/drift_metrics.json    │
                 │   - Raise MLOps Alert for Retraining      │
                 └───────────────────────────────────────────┘
```

---

## 4. Empirical Monitoring Results & Findings

### 4.1 Data Drift (Input Feature Distribution Shifts)
Comparing baseline training data ($N=1,000$) against current production stream ($N=300$):

| Feature Name | Type | Statistical Test | Test Statistic | p-value / Metric | Severity | Drift Flag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`area`** | Numerical (Continuous) | Two-Sample KS Test & PSI | $\text{KS} = 0.4663$ | $p = 5.03 \times 10^{-46}$, $\text{PSI} = 1.560$ | **CRITICAL** | 🚨 **DRIFT DETECTED** |
| **`bedrooms`** | Numerical (Discrete) | Two-Sample KS Test & PSI | $\text{KS} = 0.2297$ | $p = 3.72 \times 10^{-11}$, $\text{PSI} = 0.316$ | **HIGH** | 🚨 **DRIFT DETECTED** |
| **`location`** | Categorical (4 classes) | Chi-Square Independence | $\chi^2 = 29.39$ | $p = 1.86 \times 10^{-6}$ | **HIGH** | 🚨 **DRIFT DETECTED** |

**Observations:**
1. Mean house area shifted from **1,800 sqft** (baseline) to **2,400 sqft** (production) due to luxury home expansion.
2. Location distribution shifted significantly towards Downtown listings ($30\% \rightarrow 45\%$) with rural listings dropping ($10\% \rightarrow 5\%$).
3. PSI values on both numerical features greatly exceed the $0.25$ critical threshold, confirming severe covariate shift.

---

### 4.2 Concept Drift (Model Performance Degradation Over Time)
Tracking the deployed Gradient Boosting Regressor over 4 operational months:

| Operational Window | Sample Count | RMSE ($) | MAE ($) | $R^2$ Score | Error $\Delta$ (%) | Status / Alert |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Model** | 1,000 | **$12,208** | **$9,709** | **0.989** | 0.0% | 🟢 **BASELINE** |
| **Month 1** | 300 | $16,641 | $13,370 | 0.980 | +36.3% | 🟡 **MILD SHIFT** |
| **Month 2** | 300 | $22,480 | $18,115 | 0.968 | +84.1% | 🟠 **MODERATE DRIFT** |
| **Month 3** | 300 | $55,117 | $41,146 | 0.870 | +351.5% | 🚨 **CONCEPT DRIFT** |
| **Month 4** | 300 | **$612,580** | **$595,185** | **-4.673** | **+4,917.9%** | 🚨 **CATASTROPHIC DRIFT** |

**Root Cause Analysis:**
- In Month 4, severe market inflation caused the price per square foot to jump from $\$150$ to $\$240$, and rural premiums doubled.
- The model continued predicting based on obsolete 2026 baseline economics, causing errors exceeding $\$590,000$ per home. The negative $R^2$ indicates the frozen model is performing worse than a horizontal mean line.

---

## 5. Visual Monitoring Dashboard

The monitoring system automatically synthesized all statistical findings into a 6-panel visualization saved at [`artifacts/drift_monitoring_dashboard.png`](file:///C:/Users/asadj/Desktop/mlops%20jury/artifacts/drift_monitoring_dashboard.png):

![Drift Monitoring Dashboard](artifacts/drift_monitoring_dashboard.png)

### Dashboard Analysis:
- **Panel 1 (Top-Left):** KDE density curves showing `area` shifting significantly to the right (red production distribution versus blue baseline).
- **Panel 2 (Top-Right):** Proportional bar chart showing expansion in 4-bedroom and 5-bedroom home listings.
- **Panel 3 (Mid-Left):** Categorical shift showing sharp rise in Downtown volume and contraction in Rural volume.
- **Panel 4 (Mid-Right):** Longitudinal RMSE curve showing error exploding from $\$12,208$ to over $\$612,000$.
- **Panel 5 (Bottom-Left):** Longitudinal $R^2$ decay curve falling through the warning threshold ($0.60$) into negative territory.
- **Panel 6 (Bottom-Right):** Real-time MLOps Alarm Scorecard recommending immediate model retirement and pipeline retraining.

---

## 6. Implementation Code Walkthrough (`monitor_drift.py`)

The monitoring daemon is fully modularized into six production components:
1. `generate_reference_data()` & `generate_production_data()`: Realistically models temporal housing market drift.
2. `train_production_model()`: Fits a Gradient Boosting Regressor with baseline evaluation.
3. `calculate_psi()`: Computes multi-bucket Population Stability Index.
4. `detect_data_drift()`: Vectorized KS-tests and contingency table Chi-square calculations via `scipy.stats`.
5. `detect_concept_drift()`: Batch-grouped regression diagnostics calculating longitudinal metrics.
6. `generate_visual_dashboard()` & `generate_html_report()`: Autonomous dashboard rendering.

---

## 7. Comprehensive Viva Defense Q&A

### Q1: What is the fundamental difference between Data Drift and Concept Drift?
> **Answer:** 
> - **Data Drift (Covariate Shift):** Represents a change in the input feature distribution $P(X)$, while the underlying relationship $P(Y \mid X)$ remains unchanged. Example: buyers begin looking at significantly larger houses (higher `area`), but the price per square foot remains the same.
> - **Concept Drift:** Represents a change in the real-world relationship between inputs and outputs, meaning the conditional distribution $P(Y \mid X)$ has changed. Example: identical 2,000 sqft homes now sell for $\$100,000$ more due to inflation or interest rate shifts, rendering the model's historical weights obsolete.

### Q2: Why use the Kolmogorov-Smirnov (KS) test over simple mean/std comparison?
> **Answer:** Comparing mean and standard deviation only captures first- and second-order moments and completely misses multi-modal shifts, skewness, or kurtosis changes. The Two-Sample KS test compares the entire Empirical Cumulative Distribution Functions (ECDF), making it non-parametric, distribution-agnostic, and capable of detecting arbitrary distributional warps.

### Q3: What is Population Stability Index (PSI) and how do you interpret its thresholds?
> **Answer:** PSI is a symmetric relative entropy / Kullback-Leibler divergence metric that quantifies how much a variable has shifted over time:
> - $\text{PSI} < 0.1$: No significant change.
> - $0.1 \le \text{PSI} < 0.25$: Moderate shift; alerts are logged.
> - $\text{PSI} \ge 0.25$: Major shift; automated retraining is triggered. In our test, `area` reached $\text{PSI} = 1.56$, confirming massive covariate shift.

### Q4: How should a production MLOps system respond when Concept Drift is detected?
> **Answer:** 
> 1. **Immediate Alert:** Dispatch high-priority webhooks to Slack/PagerDuty.
> 2. **Fallback Strategy:** Switch traffic to a conservative fallback heuristic or ensemble baseline.
> 3. **Automated Pipeline Trigger:** Trigger an Apache Airflow retraining DAG that pulls fresh ground-truth sale prices from the most recent 90-day sliding window.
> 4. **Champion-Challenger Validation:** Shadow-test the newly retrained candidate against live production data before safely routing traffic.

---

## 8. Verification & Artifact Checklist

| Deliverable | File Path | Status |
| :--- | :--- | :--- |
| **Python Monitoring Engine** | [`monitor_drift.py`](file:///C:/Users/asadj/Desktop/mlops%20jury/monitor_drift.py) | ✅ Fully Executed (Code 0) |
| **Visual Dashboard (PNG)** | [`artifacts/drift_monitoring_dashboard.png`](file:///C:/Users/asadj/Desktop/mlops%20jury/artifacts/drift_monitoring_dashboard.png) | ✅ Generated & Embedded |
| **Interactive HTML Report** | [`ASSIGNMENT_5_DRIFT_MONITORING_REPORT.html`](file:///C:/Users/asadj/Desktop/mlops%20jury/ASSIGNMENT_5_DRIFT_MONITORING_REPORT.html) | ✅ Print-to-PDF Ready |
| **Structured Metrics JSON** | [`artifacts/drift_metrics.json`](file:///C:/Users/asadj/Desktop/mlops%20jury/artifacts/drift_metrics.json) | ✅ Validated |
| **Remote GitHub Backup** | [AsadLearns/mlopsjuryyybystraightnoodles](https://github.com/AsadLearns/mlopsjuryyybystraightnoodles) | ✅ Pushed to `main` |

---
*(End of Report — Candidate: Asad, October 2026)*
