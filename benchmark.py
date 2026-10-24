import os
import time
import json
import pickle
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import onnxruntime as ort

# -------------------------------------------------------------
# Stage F: Comprehensive MLOps Benchmarking Engine
# -------------------------------------------------------------
# Compares:
# 1. Native PyTorch (.pt)
# 2. Pickle-Loaded PyTorch (.pkl)
# 3. ONNX Runtime (Optimizations Enabled)
# 4. ONNX Runtime (Optimizations Disabled)
#
# Metrics Evaluated:
# - Mean & Median Latency (ms)
# - P95 & P99 Latency (ms)
# - Throughput (samples/sec)
# - Process Resident Memory (MB)
# Across Batch Sizes: B = 1, 8, 32
# -------------------------------------------------------------

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=32, num_classes=2):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        hidden = self.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_benchmark():
    print("=" * 75)
    print("       STAGE F: COMPREHENSIVE BENCHMARKING (LATENCY, THROUGHPUT, MEMORY)      ")
    print("=" * 75)

    with open("artifacts/vocab.json", "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    max_seq_len = 16

    # 1. Load PyTorch Native (.pt)
    pt_ckpt = torch.load("artifacts/model.pt", map_location="cpu", weights_only=False)
    pt_model = SentimentClassifier(vocab_size)
    pt_model.load_state_dict(pt_ckpt["model_state_dict"])
    pt_model.eval()

    # 2. Load Pickle (.pkl)
    with open("artifacts/model.pkl", "rb") as f:
        pickle_model = pickle.load(f)
        pickle_model.eval()

    # 3. Load ONNX Runtime with Graph Optimizations ENABLED
    opts_enabled = ort.SessionOptions()
    opts_enabled.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    onnx_opt_session = ort.InferenceSession("artifacts/model.onnx", opts_enabled, providers=["CPUExecutionProvider"])

    # 4. Load ONNX Runtime with Graph Optimizations DISABLED
    opts_disabled = ort.SessionOptions()
    opts_disabled.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    onnx_noopt_session = ort.InferenceSession("artifacts/model.onnx", opts_disabled, providers=["CPUExecutionProvider"])

    onnx_input_name = onnx_opt_session.get_inputs()[0].name
    onnx_output_name = onnx_opt_session.get_outputs()[0].name

    batch_sizes = [1, 8, 32]
    num_warmup = 30
    num_runs = 200

    results = []

    print(f"[*] Configuration: {num_warmup} Warm-up runs | {num_runs} Measurement runs per configuration")
    print(f"[*] Testing Batch Sizes: {batch_sizes}\n")

    targets = [
        ("PyTorch Native (.pt)", "pytorch_pt"),
        ("Pickle PyTorch (.pkl)", "pickle_pkl"),
        ("ONNX Runtime (Opt=All)", "onnx_opt"),
        ("ONNX Runtime (Opt=None)", "onnx_noopt"),
    ]

    for model_name, model_key in targets:
        print(f"[*] Benchmarking: {model_name}...")
        for b_size in batch_sizes:
            # Prepare dummy synthetic inputs
            dummy_indices = np.random.randint(1, vocab_size, size=(b_size, max_seq_len)).astype(np.int64)
            dummy_torch = torch.tensor(dummy_indices, dtype=torch.long)

            # Warm-up phase (stabilizes instruction cache and OS paging)
            for _ in range(num_warmup):
                if model_key == "pytorch_pt":
                    with torch.no_grad():
                        _ = pt_model(dummy_torch)
                elif model_key == "pickle_pkl":
                    with torch.no_grad():
                        _ = pickle_model(dummy_torch)
                elif model_key == "onnx_opt":
                    _ = onnx_opt_session.run([onnx_output_name], {onnx_input_name: dummy_indices})
                elif model_key == "onnx_noopt":
                    _ = onnx_noopt_session.run([onnx_output_name], {onnx_input_name: dummy_indices})

            # Steady-state measurement phase
            latencies = []
            start_total = time.perf_counter()
            for _ in range(num_runs):
                t0 = time.perf_counter()
                if model_key == "pytorch_pt":
                    with torch.no_grad():
                        _ = pt_model(dummy_torch)
                elif model_key == "pickle_pkl":
                    with torch.no_grad():
                        _ = pickle_model(dummy_torch)
                elif model_key == "onnx_opt":
                    _ = onnx_opt_session.run([onnx_output_name], {onnx_input_name: dummy_indices})
                elif model_key == "onnx_noopt":
                    _ = onnx_noopt_session.run([onnx_output_name], {onnx_input_name: dummy_indices})
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0) # Convert to ms

            total_elapsed = time.perf_counter() - start_total
            mem_mb = get_memory_mb()

            mean_lat = np.mean(latencies)
            median_lat = np.median(latencies)
            p95_lat = np.percentile(latencies, 95)
            throughput = (b_size * num_runs) / total_elapsed

            results.append({
                "Model / Runtime": model_name,
                "Batch Size": b_size,
                "Mean Latency (ms)": round(mean_lat, 4),
                "Median Latency (ms)": round(median_lat, 4),
                "P95 Latency (ms)": round(p95_lat, 4),
                "Throughput (samples/s)": round(throughput, 1),
                "Process Memory (MB)": round(mem_mb, 2)
            })

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = "artifacts/benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[*] Saved benchmark metrics to: {csv_path}")

    # Display Pretty Table
    print("\n" + "=" * 80)
    print(f"{'Model / Runtime':<26} | {'Batch':<5} | {'Mean (ms)':<9} | {'Median (ms)':<11} | {'Throughput':<12} | {'RAM (MB)'}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['Model / Runtime']:<26} | {row['Batch Size']:<5} | {row['Mean Latency (ms)']:<9.4f} | {row['Median Latency (ms)']:<11.4f} | {row['Throughput (samples/s)']:<12.1f} | {row['Process Memory (MB)']}")
    print("=" * 80)

    # ---------------------------------------------------------
    # Generate Visualizations (Matplotlib)
    # ---------------------------------------------------------
    print("\n[*] Generating benchmark visualization plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = df["Model / Runtime"].unique()
    x = np.arange(len(batch_sizes))
    width = 0.2

    # Plot 1: Mean Latency across Batch Sizes
    ax1 = axes[0]
    for idx, m in enumerate(models):
        subset = df[df["Model / Runtime"] == m]
        ax1.bar(x + idx * width, subset["Mean Latency (ms)"], width, label=m)

    ax1.set_xlabel("Batch Size", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Latency (ms) [Lower is Better]", fontsize=11, fontweight="bold")
    ax1.set_title("Inference Latency vs. Batch Size", fontsize=13, fontweight="bold")
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([f"B = {b}" for b in batch_sizes])
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Plot 2: Throughput across Batch Sizes
    ax2 = axes[1]
    for idx, m in enumerate(models):
        subset = df[df["Model / Runtime"] == m]
        ax2.bar(x + idx * width, subset["Throughput (samples/s)"], width, label=m)

    ax2.set_xlabel("Batch Size", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Throughput (Samples/Sec) [Higher is Better]", fontsize=11, fontweight="bold")
    ax2.set_title("Inference Throughput vs. Batch Size", fontsize=13, fontweight="bold")
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([f"B = {b}" for b in batch_sizes])
    ax2.legend(loc="upper left")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    chart_path = "artifacts/benchmark_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[*] Saved high-resolution comparison chart to: {chart_path}")
    print("\n[SUCCESS] Stage F Complete! Benchmarking and graphs ready for jury presentation.")

if __name__ == "__main__":
    run_benchmark()
