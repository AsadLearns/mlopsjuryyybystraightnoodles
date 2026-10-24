# High-Performance Model Serialization & Edge vs. Cloud Benchmark

[![GitHub Repository](https://img.shields.io/badge/GitHub-AsadLearns%2Fmlopsjuryyybystraightnoodles-blue?logo=github)](https://github.com/AsadLearns/mlopsjuryyybystraightnoodles)
**Author:** Asad ([@AsadLearns](https://github.com/AsadLearns))  
**Repository:** [https://github.com/AsadLearns/mlopsjuryyybystraightnoodles](https://github.com/AsadLearns/mlopsjuryyybystraightnoodles)

A production-grade MLOps college practical project demonstrating end-to-end model training, serialization into Pickle (`.pkl`), PyTorch (`.pt`), and ONNX (`.onnx`), and benchmarking across lightweight **Edge** (ONNX Runtime) and containerized **Cloud** (FastAPI + Docker) environments.

---

## 🚀 Quick Execution Guide

Ensure you are inside the project folder in PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 1. Train the Neural Model (Stage B)
```powershell
python train.py
```
*Generates:* `artifacts/vocab.json` and `artifacts/model_weights.pt`.

### 2. Serialize into 3 Formats & Verify Parity (Stage C)
```powershell
python serialize.py
```
*Generates:* `artifacts/model.pkl`, `artifacts/model.pt`, and `artifacts/model.onnx`. Verifies mathematical parity ($< 10^{-6}$ discrepancy).

### 3. Run Edge Inference with ONNX Runtime (Stage D)
```powershell
python edge_infer.py
```
*Runs inference with ZERO PyTorch dependency in ~0.05-0.10 ms.*

### 4. Run Cloud REST API Microservice (Stage E)
```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```
*Open interactive documentation:* `http://127.0.0.1:8000/docs`

### 5. Run Comprehensive Benchmark Suite (Stage F)
```powershell
python benchmark.py
```
*Generates:* `artifacts/benchmark_results.csv` and `artifacts/benchmark_comparison.png`.

---

## 🐳 Optional Docker Deployment
Build and run the containerized cloud service:
```powershell
docker build -t sentiment-inference:v1 .
docker run -d -p 8000:8000 --name sentiment-service sentiment-inference:v1
```

---

## 📊 Benchmark Summary (Measured on Windows CPU)

- **Batch Size 1 (Real-time user query):**
  - ONNX Runtime: **~0.039 ms** (23,800 inferences/sec)
  - PyTorch Native: **~0.227 ms** (4,348 inferences/sec)
  - **Speedup: ~5.7x lower latency with ONNX Runtime**
- **Batch Size 32 (Micro-batch processing):**
  - ONNX Runtime achieves over **300,000 samples/sec** throughput.
- **Graph Optimizations:**
  - `ORT_ENABLE_ALL` reduces latency by ~27% compared to unoptimized ONNX graphs.

---

## 📄 Complete Jury Report & Viva Questions
For complete architecture diagrams, screenshots placeholders, viva questions, and defense pitch, see:
👉 **[`PROJECT_REPORT_AND_VIVA.md`](PROJECT_REPORT_AND_VIVA.md)**
