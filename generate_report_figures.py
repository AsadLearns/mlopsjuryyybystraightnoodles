"""
Generate dedicated high-res exact-match implementation graphs for:
- Page 5: Monthly sequence length / word count drift (Reference vs Incoming)
- Page 6: Service monitoring metrics (Latency vs CPU utilization)
- Page 7: Global feature importance (SHAP token contributions)
- Page 8: Group-level fairness review (False Positive Rate vs False Negative Rate)
- Page 9: Cloud cost composition (Compute, Storage, Networking, Monitoring)
"""

import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("custom_evidence", exist_ok=True)
plt.rcParams['font.family'] = 'DejaVu Sans'

# 1. Page 5: Data Drift Distribution (Reference vs Incoming Sequence Length)
np.random.seed(42)
ref_len = np.random.normal(loc=7.0, scale=2.0, size=1000).clip(2, 16)
incoming_len = np.random.normal(loc=14.0, scale=3.5, size=1000).clip(4, 25)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(ref_len, bins=25, density=True, alpha=0.55, color='#4A90E2', label='Reference data (Mean: 7.0 words)', edgecolor='none')
ax.hist(incoming_len, bins=25, density=True, alpha=0.55, color='#E67E22', label='Simulated incoming data (Mean: 14.0 words)', edgecolor='none')
ax.set_title("Review sequence length: reference vs incoming data", fontsize=11, fontweight='normal', pad=10)
ax.set_xlabel("Review word count (sequence length)", fontsize=9.5)
ax.set_ylabel("Density", fontsize=9.5)
ax.legend(loc='upper right', frameon=True, fontsize=8.5)
ax.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("custom_evidence/fig_page5_data_drift.png", dpi=250)
plt.close()

# 2. Page 6: Service Monitoring Metrics (Latency ms vs CPU %)
intervals = np.arange(1, 13)
latency = np.array([120, 125, 130, 128, 134, 142, 150, 162, 185, 240, 310, 390])
cpu = np.array([38, 40, 42, 41, 44, 46, 49, 53, 60, 68, 77, 86])

fig, ax1 = plt.subplots(figsize=(6.8, 3.8))
ax1.plot(intervals, latency, 'ko-', label='Latency (ms)', linewidth=1.5, markersize=5)
ax1.set_xlabel("Monitoring interval", fontsize=9.5)
ax1.set_ylabel("Latency (ms)", fontsize=9.5)
ax1.set_ylim(100, 420)
ax1.grid(True, linestyle=':', alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(intervals, cpu, 's--', color='#555555', label='CPU (%)', linewidth=1.5, markersize=5)
ax2.set_ylabel("CPU utilization (%)", fontsize=9.5)
ax2.set_ylim(30, 95)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, fontsize=8.5)
plt.title("Illustrative service monitoring metrics (FastAPI + ONNX Runtime)", fontsize=10.5, pad=10)
plt.tight_layout()
plt.savefig("custom_evidence/fig_page6_service_monitoring.png", dpi=250)
plt.close()

# 3. Page 7: Global Feature Importance (SHAP Token Attributions)
features = ['terrible', 'waste', 'garbage', 'horrible', 'loved', 'delightful', 'masterpiece', 'fantastic']
importance = [0.06, 0.09, 0.13, 0.16, 0.19, 0.23, 0.27, 0.32]

fig, ax = plt.subplots(figsize=(7, 3.8))
bars = ax.barh(features, importance, color='#D3D3D3', edgecolor='#000000', height=0.65)
ax.set_xlim(0, 0.36)
ax.set_xlabel("Illustrative mean absolute SHAP contribution", fontsize=9.5)
ax.set_title("Example global feature-importance view (Sentiment Vocabulary)", fontsize=10.5, pad=10)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.008, bar.get_y() + bar.get_height()/2.0, f"{w:.2f}", ha='left', va='center', fontsize=8.5)
ax.grid(axis='x', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("custom_evidence/fig_page7_feature_importance.png", dpi=250)
plt.close()

# 4. Page 8: Group-Level Fairness Review (False-Positive Rate vs False-Negative Rate)
groups = ['Slice A (Short Texts)', 'Slice B (Medium Texts)', 'Slice C (Long Texts)']
fpr = [0.08, 0.11, 0.09]
fnr = [0.18, 0.21, 0.19]
x = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(6.8, 3.8))
rects1 = ax.bar(x - width/2, fpr, width, label='False-positive rate', color='#E0E0E0', edgecolor='#000000')
rects2 = ax.bar(x + width/2, fnr, width, label='False-negative rate', color='#9E9E9E', edgecolor='#000000')
ax.set_ylabel("Rate", fontsize=9.5)
ax.set_ylim(0, 0.26)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=9)
ax.set_title("Example group-level error-rate review (Sentiment Length Slices)", fontsize=10.5, pad=10)
ax.legend(loc='upper right', frameon=True, fontsize=8.5)
ax.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("custom_evidence/fig_page8_group_error_rate.png", dpi=250)
plt.close()

# 5. Page 9: Cloud Cost Composition
categories = ['Compute', 'Storage', 'Networking', 'Monitoring']
shares = [58, 14, 8, 20]

fig, ax = plt.subplots(figsize=(6.8, 3.8))
bars = ax.bar(categories, shares, color='#D3D3D3', edgecolor='#000000', width=0.6)
ax.set_ylabel("Illustrative share of monthly cost (%)", fontsize=9.5)
ax.set_ylim(0, 65)
ax.set_title("Example cloud cost composition (Docker + ONNX Microservice)", fontsize=10.5, pad=10)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, h + 1.2, f"{int(h)}%", ha='center', va='bottom', fontsize=9)
ax.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("custom_evidence/fig_page9_cloud_cost.png", dpi=250)
plt.close()

# 6. Page 3 Architecture Diagram
fig, ax = plt.subplots(figsize=(8.5, 2.2))
ax.axis('off')

boxes = ["Customer\nreviews", "Data validation\n& preprocessing", "Model training\n& evaluation", "Model registry\n& ONNX export", "Prediction\nservice / API", "Support\naction"]
x_coords = np.linspace(0.08, 0.92, len(boxes))
box_width = 0.12
box_height = 0.45

for i, (b, x) in enumerate(zip(boxes, x_coords)):
    rect = plt.Rectangle((x - box_width/2, 0.5), box_width, box_height, facecolor='#ffffff', edgecolor='#000000', linewidth=1)
    ax.add_patch(rect)
    ax.text(x, 0.72, b, ha='center', va='center', fontsize=8, fontfamily='DejaVu Sans')
    if i < len(boxes) - 1:
        next_x = x_coords[i+1]
        ax.annotate('', xy=(next_x - box_width/2, 0.725), xytext=(x + box_width/2, 0.725),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

# Bottom monitoring rectangle
rect_bottom = plt.Rectangle((0.25, 0.05), 0.50, 0.30, facecolor='#ffffff', edgecolor='#000000', linewidth=1)
ax.add_patch(rect_bottom)
ax.text(0.5, 0.20, "Monitoring, logs, drift checks, explainability, security and alerts", ha='center', va='center', fontsize=8, fontfamily='DejaVu Sans')

# Connecting lines
ax.plot([0.5, 0.5], [0.5, 0.35], 'k-', lw=1)
ax.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle="->", color="black", lw=1))

plt.title("Proposed MLOps architecture for sentiment analysis prediction", fontsize=9.5, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig("custom_evidence/fig_page3_architecture.png", dpi=250)
plt.close()

# 7. Page 8 Secure ML Pipeline Controls Diagram
fig, ax = plt.subplots(figsize=(8.5, 2.2))
ax.axis('off')

sec_boxes = ["Identity &\nleast privilege", "Validate data\n& scan inputs", "Protect secrets\n& encrypt data", "Version and verify\nmodel artifact", "Approve and audit\ndeployment"]
x_coords = np.linspace(0.1, 0.9, len(sec_boxes))
box_width = 0.13
box_height = 0.45

for i, (b, x) in enumerate(zip(sec_boxes, x_coords)):
    rect = plt.Rectangle((x - box_width/2, 0.45), box_width, box_height, facecolor='#ffffff', edgecolor='#000000', linewidth=1)
    ax.add_patch(rect)
    ax.text(x, 0.675, b, ha='center', va='center', fontsize=8, fontfamily='DejaVu Sans')
    if i < len(sec_boxes) - 1:
        next_x = x_coords[i+1]
        ax.annotate('', xy=(next_x - box_width/2, 0.675), xytext=(x + box_width/2, 0.675),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

ax.text(0.5, 0.18, "Audit logs • role-based access • TLS • dependency checks • rollback plan", ha='center', va='center', fontsize=8, fontfamily='DejaVu Sans')
plt.title("Secure ML pipeline controls", fontsize=9.5, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig("custom_evidence/fig_page8_secure_pipeline.png", dpi=250)
plt.close()

print("✅ All dedicated figure screenshots created in custom_evidence/!")
