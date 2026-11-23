import re

report_path = r'C:\Users\asadj\Desktop\mlops jury\ADVANCED_MLOPS_REPORTS_FORMAT_MATCH.html'
with open(report_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Update CSS for figure images
old_fig_css = """.figure img {
    max-width: 98%;
    border: 1px solid #777777;
  }"""
new_fig_css = """.figure img {
    max-width: 95%;
    max-height: 360px;
    object-fit: contain;
    border: 1px solid #777777;
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }"""
if old_fig_css in html:
    html = html.replace(old_fig_css, new_fig_css)
    print('Updated figure CSS!')

appendix_html = """
<div class="page-break"></div>

<!-- ==================== APPENDIX: IMPLEMENTATION SCREENSHOTS ==================== -->
<div class="header-tag">ADVANCED MLOps PROJECT REPORT &bull; IMPLEMENTATION SCREENSHOTS GALLERY</div>

<h2>Appendix: Live Implementation &amp; Execution Screenshots</h2>
<p>This gallery presents actual execution screenshots captured directly from our operational environment, demonstrating live verification of all MLOps components including automated testing, drift monitoring, interactive FastAPI serving, CI/CD pipelines, and Airflow orchestration DAGs.</p>

<div class="figure">
  <img src="custom_evidence/ss_terminal_pytest_all_25.png" alt="Automated Test Suite Pytest Execution">
  <div class="caption">Figure A1: Automated Pytest Execution Suite &mdash; All 25 Test Cases (TC-01 through TC-25) executing and passing in 7.17s.</div>
</div>

<div class="figure">
  <img src="custom_evidence/ss_terminal_monitor_drift.png" alt="Drift Detection Engine Execution">
  <div class="caption">Figure A2: Live Drift Monitoring Engine Terminal Execution (<code>python monitor_drift.py</code>) with Kolmogorov-Smirnov statistical testing and alert trigger.</div>
</div>

<div class="page-footer">High-Performance Sentiment Analysis | Academic Project Report</div>

<div class="page-break"></div>

<div class="header-tag">ADVANCED MLOps PROJECT REPORT &bull; IMPLEMENTATION SCREENSHOTS GALLERY</div>

<div class="figure">
  <img src="artifacts/drift_monitoring_dashboard.png" alt="Real-Time Drift Monitoring Dashboard">
  <div class="caption">Figure A3: Visual Multi-Metric Drift Monitoring Dashboard &mdash; Tracking feature length drift, p-values, rolling F1-score, and retraining trigger across production serving days.</div>
</div>

<div class="figure">
  <img src="custom_evidence/ev_02_model_evaluation_confusion_matrix.png" alt="Model Evaluation and Confusion Matrix">
  <div class="caption">Figure A4: Model Performance Evaluation &mdash; Confusion Matrix and Classification Report verifying 100% accuracy on our sentiment dataset.</div>
</div>

<div class="page-footer">High-Performance Sentiment Analysis | Academic Project Report</div>

<div class="page-break"></div>

<div class="header-tag">ADVANCED MLOps PROJECT REPORT &bull; IMPLEMENTATION SCREENSHOTS GALLERY</div>

<div class="figure">
  <img src="screenshots/5_swagger_docs_health.png" alt="FastAPI Interactive Swagger Documentation">
  <div class="caption">Figure A5: Production FastAPI Microservice &mdash; Interactive Swagger OpenAPI Documentation with health checks, Prometheus /metrics, and RBAC admin endpoints.</div>
</div>

<div class="figure">
  <img src="screenshots/6_github_actions_all_workflows.png" alt="GitHub Actions CI/CD Automated Pipelines">
  <div class="caption">Figure A6: GitHub Actions Continuous Integration and Deployment &mdash; Automated test runs, linting, Docker build, and staging deployments passing on commit.</div>
</div>

<div class="page-footer">High-Performance Sentiment Analysis | Academic Project Report</div>

<div class="page-break"></div>

<div class="header-tag">ADVANCED MLOps PROJECT REPORT &bull; IMPLEMENTATION SCREENSHOTS GALLERY</div>

<div class="figure">
  <img src="screenshots/10_airflow_dag_graph_success.png" alt="Apache Airflow Model Retraining DAG">
  <div class="caption">Figure A7: Apache Airflow Automated Orchestration &mdash; Successful execution graph of the sentiment model retraining and validation pipeline DAG.</div>
</div>

<div class="figure">
  <img src="custom_evidence/ev_06_shap_explainability.png" alt="SHAP Token-Level Explainability">
  <div class="caption">Figure A8: Explainable AI (XAI) &mdash; SHAP Token Attribution and Force Plot illustrating positive and negative word contributions to predictions.</div>
</div>

<div class="page-footer">High-Performance Sentiment Analysis | Academic Project Report</div>
"""

if 'Appendix: Live Implementation' not in html:
    html = html.replace('</body>', appendix_html + '\n</body>')
    print('Added Appendix gallery!')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html)
print('Successfully written updated report with Appendix Screenshots Gallery!')
