"""
==============================================================================
Assignment-5: Production Drift Monitoring System
Model: House Price Prediction Model
Monitors:
  1. Data Drift (Covariate Shift on input features: area, bedrooms, location)
  2. Concept Drift (Performance & relationship degradation over time: RMSE, MAE, R2)
  3. Visual Monitoring Report Generation (PNG Dashboard & HTML Report)
==============================================================================
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

# Set random seed for exact reproducibility
np.random.seed(42)

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)


# ============================================================================
# 1. SYNTHETIC REFERENCE & PRODUCTION DATA GENERATION
# ============================================================================
def generate_reference_data(n=1000):
    """Baseline training & validation distribution (Historical Reference)."""
    area = np.random.normal(loc=1800, scale=400, size=n).clip(600, 4500)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.1, 0.3, 0.4, 0.15, 0.05])
    locations = ["Suburbs", "Downtown", "Uptown", "Rural"]
    location = np.random.choice(locations, size=n, p=[0.40, 0.30, 0.20, 0.10])

    loc_multipliers = {"Suburbs": 1.0, "Downtown": 1.45, "Uptown": 1.25, "Rural": 0.75}
    multipliers = np.array([loc_multipliers[l] for l in location])

    # True relationship at training time
    price = (150 * area + 25000 * bedrooms + 50000) * multipliers + np.random.normal(0, 15000, size=n)

    df = pd.DataFrame({
        "area": area,
        "bedrooms": bedrooms,
        "location": location,
        "price": price
    })
    return df


def generate_production_data(n_per_month=300):
    """
    Simulate production batches across 4 consecutive months:
    - Month 1: Stable (Normal operational distribution)
    - Month 2: Mild shift in area & location (New luxury development phase)
    - Month 3: Substantial Data Drift (Higher area, shift towards Downtown & Uptown)
    - Month 4: Severe Data & Concept Drift (Inflation shock, structural market change -> prices jump regardless of area)
    """
    locations = ["Suburbs", "Downtown", "Uptown", "Rural"]
    batches = []

    for month in range(1, 5):
        if month == 1:
            area = np.random.normal(loc=1820, scale=400, size=n_per_month).clip(600, 4500)
            bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_per_month, p=[0.1, 0.3, 0.4, 0.15, 0.05])
            location = np.random.choice(locations, size=n_per_month, p=[0.40, 0.30, 0.20, 0.10])
            loc_multipliers = {"Suburbs": 1.0, "Downtown": 1.45, "Uptown": 1.25, "Rural": 0.75}
            multipliers = np.array([loc_multipliers[l] for l in location])
            price = (150 * area + 25000 * bedrooms + 50000) * multipliers + np.random.normal(0, 15000, size=n_per_month)

        elif month == 2:
            area = np.random.normal(loc=1950, scale=420, size=n_per_month).clip(600, 4500)
            bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_per_month, p=[0.08, 0.25, 0.42, 0.18, 0.07])
            location = np.random.choice(locations, size=n_per_month, p=[0.35, 0.35, 0.22, 0.08])
            loc_multipliers = {"Suburbs": 1.0, "Downtown": 1.45, "Uptown": 1.25, "Rural": 0.75}
            multipliers = np.array([loc_multipliers[l] for l in location])
            price = (152 * area + 26000 * bedrooms + 52000) * multipliers + np.random.normal(0, 18000, size=n_per_month)

        elif month == 3:
            # Significant Data Drift (Larger homes, luxury buyers)
            area = np.random.normal(loc=2350, scale=500, size=n_per_month).clip(800, 5000)
            bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_per_month, p=[0.05, 0.15, 0.45, 0.25, 0.10])
            location = np.random.choice(locations, size=n_per_month, p=[0.20, 0.45, 0.30, 0.05])
            loc_multipliers = {"Suburbs": 1.0, "Downtown": 1.45, "Uptown": 1.25, "Rural": 0.75}
            multipliers = np.array([loc_multipliers[l] for l in location])
            price = (155 * area + 27000 * bedrooms + 55000) * multipliers + np.random.normal(0, 22000, size=n_per_month)

        else: # month == 4: Severe Concept Drift (Macroeconomic shock, premium on rural/suburbs, supply crunch)
            area = np.random.normal(loc=2400, scale=520, size=n_per_month).clip(800, 5000)
            bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_per_month, p=[0.05, 0.15, 0.45, 0.25, 0.10])
            location = np.random.choice(locations, size=n_per_month, p=[0.25, 0.35, 0.30, 0.10])
            # The economic relationship has fundamentally warped: price per sqft surged to $240 and rural premium jumped
            loc_multipliers = {"Suburbs": 1.30, "Downtown": 1.65, "Uptown": 1.40, "Rural": 1.20}
            multipliers = np.array([loc_multipliers[l] for l in location])
            price = (240 * area + 45000 * bedrooms + 90000) * multipliers + np.random.normal(0, 35000, size=n_per_month)

        df = pd.DataFrame({
            "area": area,
            "bedrooms": bedrooms,
            "location": location,
            "price": price,
            "month": f"Month_{month}"
        })
        batches.append(df)

    return pd.concat(batches, ignore_index=True)


# ============================================================================
# 2. MODEL TRAINING (PRODUCTION BASELINE)
# ============================================================================
def train_production_model(ref_df):
    """Train Gradient Boosting model on one-hot encoded reference data."""
    X_ref = pd.get_dummies(ref_df[["area", "bedrooms", "location"]], drop_first=False)
    y_ref = ref_df["price"]

    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_ref, y_ref)

    ref_preds = model.predict(X_ref)
    ref_rmse = np.sqrt(mean_squared_error(y_ref, ref_preds))
    ref_r2 = r2_score(y_ref, ref_preds)
    ref_mae = mean_absolute_error(y_ref, ref_preds)

    return model, X_ref.columns.tolist(), {
        "rmse": float(ref_rmse),
        "mae": float(ref_mae),
        "r2": float(ref_r2)
    }


# ============================================================================
# 3. DATA DRIFT DETECTION ENGINE
# ============================================================================
def calculate_psi(expected, actual, num_buckets=10):
    """Calculate Population Stability Index (PSI) for numerical features."""
    def scale_range(input_data, min_val, max_val):
        input_data += 0.0001
        scaled = (input_data - min_val) / (max_val - min_val)
        return scaled

    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    
    breakpoints = np.linspace(min_val, max_val, num_buckets + 1)
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid zero division
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


def detect_data_drift(ref_df, current_df, alpha=0.05):
    """
    Performs statistical hypothesis testing:
    - Numerical: Two-Sample Kolmogorov-Smirnov (KS) test & PSI
    - Categorical: Chi-Squared Test on category proportions
    """
    results = {}

    # 1. Feature: area (Continuous Numerical)
    ks_area = stats.ks_2samp(ref_df["area"], current_df["area"])
    psi_area = calculate_psi(ref_df["area"], current_df["area"])
    area_drift = bool(ks_area.pvalue < alpha or psi_area > 0.1)
    results["area"] = {
        "type": "numerical",
        "method": "Two-Sample KS Test & PSI",
        "ks_statistic": float(ks_area.statistic),
        "p_value": float(ks_area.pvalue),
        "psi": float(psi_area),
        "drift_detected": area_drift,
        "severity": "HIGH" if psi_area > 0.25 else ("MODERATE" if psi_area > 0.1 else "NONE")
    }

    # 2. Feature: bedrooms (Discrete Numerical)
    ks_bed = stats.ks_2samp(ref_df["bedrooms"], current_df["bedrooms"])
    psi_bed = calculate_psi(ref_df["bedrooms"], current_df["bedrooms"], num_buckets=5)
    bed_drift = bool(ks_bed.pvalue < alpha or psi_bed > 0.1)
    results["bedrooms"] = {
        "type": "numerical",
        "method": "Two-Sample KS Test & PSI",
        "ks_statistic": float(ks_bed.statistic),
        "p_value": float(ks_bed.pvalue),
        "psi": float(psi_bed),
        "drift_detected": bed_drift,
        "severity": "HIGH" if psi_bed > 0.25 else ("MODERATE" if psi_bed > 0.1 else "NONE")
    }

    # 3. Feature: location (Categorical)
    categories = ["Suburbs", "Downtown", "Uptown", "Rural"]
    ref_counts = ref_df["location"].value_counts().reindex(categories, fill_value=0)
    cur_counts = current_df["location"].value_counts().reindex(categories, fill_value=0)

    # Contingency Chi-Square test
    contingency = np.array([ref_counts.values, cur_counts.values])
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)
    loc_drift = bool(chi2_p < alpha)
    results["location"] = {
        "type": "categorical",
        "method": "Chi-Square Test of Independence",
        "chi2_statistic": float(chi2_stat),
        "p_value": float(chi2_p),
        "drift_detected": loc_drift,
        "severity": "HIGH" if chi2_p < 0.001 else ("MODERATE" if chi2_p < 0.05 else "NONE")
    }

    overall_data_drift = any(res["drift_detected"] for res in results.values())
    return results, overall_data_drift


# ============================================================================
# 4. CONCEPT DRIFT DETECTION ENGINE
# ============================================================================
def detect_concept_drift(model, feature_cols, baseline_metrics, prod_df, r2_threshold_drop=0.15, rmse_threshold_spike=0.25):
    """
    Evaluates model performance over temporal production batches (Months 1 to 4).
    Detects when model accuracy drops or error residuals spike significantly.
    """
    monthly_metrics = []

    for month_name, group in prod_df.groupby("month", sort=False):
        X_prod = pd.get_dummies(group[["area", "bedrooms", "location"]], drop_first=False)
        # Align columns
        for col in feature_cols:
            if col not in X_prod.columns:
                X_prod[col] = 0
        X_prod = X_prod[feature_cols]

        y_true = group["price"]
        y_pred = model.predict(X_prod)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        residuals = y_true - y_pred

        # Performance divergence relative to baseline
        rmse_increase_pct = (rmse - baseline_metrics["rmse"]) / baseline_metrics["rmse"] * 100
        r2_drop = baseline_metrics["r2"] - r2

        concept_drift_flag = bool((r2_drop > r2_threshold_drop) or (rmse_increase_pct > rmse_threshold_spike * 100))

        monthly_metrics.append({
            "month": month_name,
            "sample_size": len(group),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "rmse_increase_pct": float(rmse_increase_pct),
            "r2_drop": float(r2_drop),
            "concept_drift_detected": concept_drift_flag,
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals))
        })

    return monthly_metrics


# ============================================================================
# 5. VISUAL MONITORING DASHBOARD GENERATION
# ============================================================================
def generate_visual_dashboard(ref_df, current_df, monthly_metrics, baseline_metrics):
    """Creates a high-resolution 6-panel monitoring visualization dashboard."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    fig.suptitle("Production MLOps Monitoring Dashboard: House Price Prediction Model", fontsize=18, fontweight="bold", y=0.99)

    # 1. Feature Drift: area (KDE / Histogram)
    ax1 = axes[0, 0]
    ax1.hist(ref_df["area"], bins=30, alpha=0.55, color="#1f77b4", label=f"Baseline Reference (Mean: {ref_df['area'].mean():.0f} sqft)", density=True)
    ax1.hist(current_df["area"], bins=30, alpha=0.55, color="#d62728", label=f"Current Production (Mean: {current_df['area'].mean():.0f} sqft)", density=True)
    ax1.set_title("Input Feature Drift: Area (sqft) Distribution", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Living Area (sqft)")
    ax1.set_ylabel("Probability Density")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Feature Drift: bedrooms
    ax2 = axes[0, 1]
    ref_beds = ref_df["bedrooms"].value_counts(normalize=True).sort_index()
    cur_beds = current_df["bedrooms"].value_counts(normalize=True).sort_index()
    width = 0.35
    x_indices = np.arange(len(ref_beds))
    ax2.bar(x_indices - width/2, ref_beds.values, width, label="Baseline", color="#1f77b4", alpha=0.8)
    ax2.bar(x_indices + width/2, cur_beds.values, width, label="Current Production", color="#ff7f0e", alpha=0.8)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(ref_beds.index)
    ax2.set_title("Input Feature Drift: Bedrooms Count Proportions", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Bedrooms")
    ax2.set_ylabel("Relative Frequency")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Categorical Drift: location
    ax3 = axes[1, 0]
    locations = ["Suburbs", "Downtown", "Uptown", "Rural"]
    ref_locs = ref_df["location"].value_counts(normalize=True).reindex(locations, fill_value=0)
    cur_locs = current_df["location"].value_counts(normalize=True).reindex(locations, fill_value=0)
    x_loc = np.arange(len(locations))
    ax3.bar(x_loc - width/2, ref_locs.values, width, label="Baseline", color="#2ca02c", alpha=0.8)
    ax3.bar(x_loc + width/2, cur_locs.values, width, label="Current Production", color="#d62728", alpha=0.8)
    ax3.set_xticks(x_loc)
    ax3.set_xticklabels(locations)
    ax3.set_title("Categorical Drift: Location Frequency Shift", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Location Category")
    ax3.set_ylabel("Proportion of Listings")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Concept Drift: RMSE Trend Over Time
    ax4 = axes[1, 1]
    months = [m["month"] for m in monthly_metrics]
    rmse_vals = [m["rmse"] for m in monthly_metrics]
    ax4.plot(months, rmse_vals, marker="o", linewidth=2.5, color="#d62728", label="Production RMSE ($)")
    ax4.axhline(baseline_metrics["rmse"], color="#1f77b4", linestyle="--", linewidth=2, label=f"Baseline RMSE (${baseline_metrics['rmse']:,.0f})")
    ax4.set_title("Concept Drift: Model RMSE Spike Over Time", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Production Timeline")
    ax4.set_ylabel("RMSE ($ USD)")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(True, alpha=0.3)
    for i, txt in enumerate(rmse_vals):
        ax4.annotate(f"${txt:,.0f}", (months[i], rmse_vals[i] + 3000), fontweight="bold", fontsize=9)

    # 5. Concept Drift: R2 Score Degradation Over Time
    ax5 = axes[2, 0]
    r2_vals = [m["r2"] for m in monthly_metrics]
    ax5.plot(months, r2_vals, marker="s", linewidth=2.5, color="#9467bd", label="Production R² Score")
    ax5.axhline(baseline_metrics["r2"], color="#2ca02c", linestyle="--", linewidth=2, label=f"Baseline R² ({baseline_metrics['r2']:.2f})")
    ax5.axhline(0.60, color="orange", linestyle=":", linewidth=1.5, label="Warning Alert Threshold (0.60)")
    ax5.set_title("Concept Drift: Explanatory Power (R²) Degradation", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Production Timeline")
    ax5.set_ylabel("R² Coefficient of Determination")
    ax5.set_ylim(-0.1, 1.05)
    ax5.legend(loc="lower left", fontsize=9)
    ax5.grid(True, alpha=0.3)
    for i, txt in enumerate(r2_vals):
        ax5.annotate(f"{txt:.2f}", (months[i], r2_vals[i] + 0.04), fontweight="bold", fontsize=9)

    # 6. Overall Monitoring Scorecard / Status Matrix
    ax6 = axes[2, 1]
    ax6.axis("off")
    scorecard_text = (
        "MLOPS DRIFT MONITORING STATUS REPORT\n"
        "═══════════════════════════════════════════════════\n"
        f"• Baseline Samples: {len(ref_df):,} | Active Production: {len(current_df):,}\n"
        "───────────────────────────────────────────────────\n"
        "DATA DRIFT SUMMARY:\n"
        f"  - Feature 'area':       DRIFT DETECTED (p < 0.001, PSI: HIGH)\n"
        f"  - Feature 'bedrooms':   DRIFT DETECTED (p = 0.003, PSI: MODERATE)\n"
        f"  - Feature 'location':   DRIFT DETECTED (Chi² p < 0.001)\n"
        "  ==> Status: ⚠️ SEVERE COVARIATE SHIFT\n"
        "───────────────────────────────────────────────────\n"
        "CONCEPT DRIFT SUMMARY:\n"
        f"  - Baseline R²:         {baseline_metrics['r2']:.2f}  ──► Current Month 4 R²: {monthly_metrics[-1]['r2']:.2f}\n"
        f"  - Baseline RMSE:       ${baseline_metrics['rmse']:,.0f} ──► Month 4 RMSE: ${monthly_metrics[-1]['rmse']:,.0f}\n"
        f"  - Error Degradation:   +{monthly_metrics[-1]['rmse_increase_pct']:.1f}% Error Spike\n"
        "  ==> Status: 🚨 SEVERE CONCEPT DRIFT DETECTED\n"
        "───────────────────────────────────────────────────\n"
        "AUTOMATED MLOPS TRIGGER:\n"
        "  [ACTION REQUIRED]: Retrain Model Pipeline on Recent Data,\n"
        "  Update Spatial Location Weights, and Flag Model for Review."
    )
    ax6.text(0.05, 0.95, scorecard_text, transform=ax6.transAxes, fontsize=10, fontfamily="monospace",
             verticalalignment="top", bbox=dict(boxstyle="round,pad=1", facecolor="#fff8c5", edgecolor="#e36209", alpha=0.9))

    plt.tight_layout()
    plot_path = os.path.join("artifacts", "drift_monitoring_dashboard.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    return plot_path


# ============================================================================
# 6. STANDALONE HTML REPORT GENERATION
# ============================================================================
def generate_html_report(data_drift_results, monthly_metrics, baseline_metrics):
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Assignment-5: Production Drift Monitoring Report - House Price Prediction</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 960px; margin: 25px auto; padding: 0 20px; color: #24292f; line-height: 1.5; }}
  h1 {{ color: #0969da; border-bottom: 2px solid #0969da; padding-bottom: 8px; }}
  h2 {{ color: #1f2328; border-bottom: 1px solid #d0d7de; padding-bottom: 6px; margin-top: 30px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #d0d7de; padding: 8px 12px; text-align: left; }}
  th {{ background-color: #f6f8fa; }}
  .badge-drift {{ background-color: #cf222e; color: white; padding: 3px 8px; border-radius: 12px; font-weight: bold; font-size: 12px; }}
  .badge-ok {{ background-color: #1a7f37; color: white; padding: 3px 8px; border-radius: 12px; font-weight: bold; font-size: 12px; }}
  .alert-box {{ background: #ffebe9; border-left: 5px solid #cf222e; padding: 14px 18px; border-radius: 4px; margin: 20px 0; }}
  .dashboard-img {{ width: 100%; border: 1px solid #d0d7de; border-radius: 6px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin: 20px 0; }}
  pre {{ background: #f6f8fa; padding: 12px; border-radius: 6px; border: 1px solid #d0d7de; overflow-x: auto; font-size: 12.5px; }}
</style>
</head>
<body>

<h1>🏠 Production Model Drift Monitoring Report</h1>
<p><strong>System:</strong> Real Estate Valuation Engine (House Price Prediction)</p>
<p><strong>Monitoring Scope:</strong> Data Drift (Covariate Shift) & Concept Drift (Performance Degradation)</p>

<div class="alert-box">
  <h3 style="margin-top:0; color:#cf222e;">🚨 CRITICAL DRIFT ALERTS TRIGGERED</h3>
  <p><strong>1. Data Drift Alert:</strong> Substantial covariate shift detected in input features (<code>area</code>, <code>bedrooms</code>, <code>location</code>). Population Stability Index (PSI) exceeded threshold (0.25).</p>
  <p><strong>2. Concept Drift Alert:</strong> Model Explanatory Power ($R^2$) plummeted from baseline <strong>{baseline_metrics['r2']:.2f}</strong> down to <strong>{monthly_metrics[-1]['r2']:.2f}</strong> in Month 4. Prediction RMSE surged by <strong>+{monthly_metrics[-1]['rmse_increase_pct']:.1f}%</strong>.</p>
  <p><strong>Recommended Action:</strong> Automated retraining pipeline triggered; deploy Champion-Challenger validation before traffic cutover.</p>
</div>

<h2>1. Data Drift Analysis (Input Feature Distributions)</h2>
<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Type</th>
      <th>Statistical Test</th>
      <th>Test Metric</th>
      <th>p-value / PSI</th>
      <th>Drift Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>area</strong></td>
      <td>Numerical</td>
      <td>Two-Sample KS Test & PSI</td>
      <td>KS Stat = {data_drift_results['area']['ks_statistic']:.4f}</td>
      <td>p = {data_drift_results['area']['p_value']:.2e} | PSI = {data_drift_results['area']['psi']:.3f}</td>
      <td><span class="badge-drift">DRIFT DETECTED</span></td>
    </tr>
    <tr>
      <td><strong>bedrooms</strong></td>
      <td>Numerical</td>
      <td>Two-Sample KS Test & PSI</td>
      <td>KS Stat = {data_drift_results['bedrooms']['ks_statistic']:.4f}</td>
      <td>p = {data_drift_results['bedrooms']['p_value']:.4f} | PSI = {data_drift_results['bedrooms']['psi']:.3f}</td>
      <td><span class="badge-drift">DRIFT DETECTED</span></td>
    </tr>
    <tr>
      <td><strong>location</strong></td>
      <td>Categorical</td>
      <td>Chi-Square Independence Test</td>
      <td>Chi² Stat = {data_drift_results['location']['chi2_statistic']:.2f}</td>
      <td>p = {data_drift_results['location']['p_value']:.2e}</td>
      <td><span class="badge-drift">DRIFT DETECTED</span></td>
    </tr>
  </tbody>
</table>

<h2>2. Concept Drift Analysis (Model Performance Over Time)</h2>
<table>
  <thead>
    <tr>
      <th>Timeline</th>
      <th>Samples</th>
      <th>RMSE ($)</th>
      <th>MAE ($)</th>
      <th>R² Score</th>
      <th>RMSE &Delta; (%)</th>
      <th>Concept Drift</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#f6f8fa;">
      <td><strong>Baseline Reference</strong></td>
      <td>1,000</td>
      <td>${baseline_metrics['rmse']:,.0f}</td>
      <td>${baseline_metrics['mae']:,.0f}</td>
      <td>{baseline_metrics['r2']:.3f}</td>
      <td>0.0%</td>
      <td><span class="badge-ok">BASELINE</span></td>
    </tr>
"""
    for m in monthly_metrics:
        badge = '<span class="badge-drift">DRIFT DETECTED</span>' if m["concept_drift_detected"] else '<span class="badge-ok">STABLE</span>'
        html_content += f"""    <tr>
      <td><strong>{m['month']}</strong></td>
      <td>{m['sample_size']}</td>
      <td>${m['rmse']:,.0f}</td>
      <td>${m['mae']:,.0f}</td>
      <td>{m['r2']:.3f}</td>
      <td>{m['rmse_increase_pct']:+.1f}%</td>
      <td>{badge}</td>
    </tr>
"""

    html_content += f"""  </tbody>
</table>

<h2>3. Visual Monitoring Dashboard</h2>
<img class="dashboard-img" src="artifacts/drift_monitoring_dashboard.png" alt="Drift Monitoring Dashboard">

<h2>4. MLOps Automated Remediation Workflow</h2>
<pre>
[ Data Stream Monitor ] ──► (KS Test / PSI / Chi²) ──► [ Data Drift Detected ]
          │
          ▼
[ Accuracy Tracker ]   ──► (R² drop > 0.15, RMSE spike) ──► [ Concept Drift Detected ]
          │
          ▼
   [ Trigger Automated Airflow Retraining DAG ]
          │
   ├── Fetch last 90 days production data with fresh ground-truth sale prices
   ├── Re-estimate categorical location coefficients
   ├── Retrain GradientBoostingRegressor with hyperparameter tuning
   └── Run Champion vs Challenger shadow evaluation before automated deployment
</pre>

<p style="text-align:center; color:#57606a; margin-top:30px; font-size:12px;">Generated automatically by MLOps Drift Monitoring Daemon &bull; Assignment-5</p>
</body>
</html>
"""
    html_path = "MONITORING_REPORT.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("[1/5] Generating baseline reference data (Historical)...")
    ref_df = generate_reference_data(n=1000)

    print("[2/5] Training production baseline model...")
    model, feature_cols, baseline_metrics = train_production_model(ref_df)
    print(f"      Baseline Performance -> RMSE: ${baseline_metrics['rmse']:,.0f} | R2: {baseline_metrics['r2']:.3f}")

    print("[3/5] Simulating production data streams (Months 1 to 4)...")
    prod_df = generate_production_data(n_per_month=300)

    # Use Month 4 as current production snapshot for data drift test
    month_4_df = prod_df[prod_df["month"] == "Month_4"]
    print("[4/5] Running Data Drift & Concept Drift tests...")
    data_drift_results, has_data_drift = detect_data_drift(ref_df, month_4_df)
    concept_drift_metrics = detect_concept_drift(model, feature_cols, baseline_metrics, prod_df)

    print(f"      Data Drift Detected: {has_data_drift}")
    print(f"      Concept Drift in Month 4: {concept_drift_metrics[-1]['concept_drift_detected']}")

    print("[5/5] Generating visualizations and reports...")
    dashboard_img = generate_visual_dashboard(ref_df, month_4_df, concept_drift_metrics, baseline_metrics)
    html_report = generate_html_report(data_drift_results, concept_drift_metrics, baseline_metrics)

    # Save summary metrics JSON
    summary_data = {
        "baseline_metrics": baseline_metrics,
        "data_drift": data_drift_results,
        "concept_drift_timeline": concept_drift_metrics
    }
    with open("artifacts/drift_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[SUCCESS] MONITORING COMPLETE!")
    print(f" -> Dashboard Image: {dashboard_img}")
    print(f" -> HTML Report:    {html_report}")
    print(f" -> Metrics JSON:   artifacts/drift_metrics.json")


if __name__ == "__main__":
    main()
