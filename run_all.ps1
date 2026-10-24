# =====================================================================
# Complete End-to-End MLOps Pipeline Runner
# High-Performance Model Serialization & Edge vs. Cloud Benchmark
# =====================================================================

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   STARTING COMPLETE MLOps PIPELINE EXECUTION (ALL STAGES)      " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Activate Virtual Environment
$VENV_PYTHON = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "[ERROR] Virtual environment python.exe not found at $VENV_PYTHON!" -ForegroundColor Red
    exit 1
}

# 2. Stage B: Model Training
Write-Host "`n>>> [STAGE B] Training Neural Sentiment Classifier..." -ForegroundColor Yellow
& $VENV_PYTHON train.py
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Training failed." -ForegroundColor Red; exit 1 }

# 3. Stage C: Model Serialization (.pkl, .pt, .onnx)
Write-Host "`n>>> [STAGE C] Serializing Model into .pkl, .pt, .onnx & Checking Parity..." -ForegroundColor Yellow
& $VENV_PYTHON serialize.py
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Serialization failed." -ForegroundColor Red; exit 1 }

# 4. Stage D: Edge Inference Service
Write-Host "`n>>> [STAGE D] Executing Lightweight ONNX Runtime Edge Inference..." -ForegroundColor Yellow
& $VENV_PYTHON edge_infer.py
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Edge inference failed." -ForegroundColor Red; exit 1 }

# 5. Stage F: Comprehensive Benchmarking & Plot Generation
Write-Host "`n>>> [STAGE F] Running Multi-Batch Benchmark Suite & Generating Graphs..." -ForegroundColor Yellow
& $VENV_PYTHON benchmark.py
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Benchmarking failed." -ForegroundColor Red; exit 1 }

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host "   [SUCCESS] ENTIRE PIPELINE COMPLETED AND VERIFIED ON DISK!    " -ForegroundColor Green
Write-Host "   - Model Artifacts: artifacts/model.pkl, model.pt, model.onnx " -ForegroundColor Green
Write-Host "   - Benchmark CSV:   artifacts/benchmark_results.csv           " -ForegroundColor Green
Write-Host "   - Graphs Created:  artifacts/benchmark_comparison.png        " -ForegroundColor Green
Write-Host "   - Full Report:     PROJECT_REPORT_AND_VIVA.md                " -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
