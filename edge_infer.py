import json
import time
import numpy as np
import onnxruntime as ort

# -------------------------------------------------------------
# Edge Inference Service (Zero PyTorch Dependency)
# -------------------------------------------------------------
# In a real edge deployment (e.g. smart IoT device, mobile app),
# installing a 1GB+ deep learning library like PyTorch is impractical.
# ONNX Runtime provides a tiny C++ runtime with pure Python/NumPy bindings.
# -------------------------------------------------------------

class EdgeSentimentClassifier:
    def __init__(self, onnx_model_path="artifacts/model.onnx", vocab_path="artifacts/vocab.json"):
        # Load Vocabulary
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.max_seq_len = 16

        # Initialize ONNX Runtime Session (CPU Execution Provider)
        session_options = ort.SessionOptions()
        # Enable basic and extended graph optimizations (constant folding, operator fusion)
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_model_path, session_options, providers=["CPUExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.labels = {0: "NEGATIVE", 1: "POSITIVE"}

    def preprocess(self, text):
        tokens = text.lower().split()
        indices = [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in tokens]
        if len(indices) < self.max_seq_len:
            indices += [self.vocab["<PAD>"]] * (self.max_seq_len - len(indices))
        else:
            indices = indices[:self.max_seq_len]
        return np.array([indices], dtype=np.int64)

    def predict(self, text):
        input_tensor = self.preprocess(text)

        # Measure low-overhead inference latency
        t0 = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        logits = outputs[0][0]
        # Softmax probability calculation
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        predicted_class = int(np.argmax(probabilities))

        return {
            "sentence": text,
            "sentiment": self.labels[predicted_class],
            "confidence": float(probabilities[predicted_class] * 100.0),
            "latency_ms": latency_ms,
            "probabilities": {
                "NEGATIVE": float(probabilities[0] * 100.0),
                "POSITIVE": float(probabilities[1] * 100.0)
            }
        }

def main():
    print("=" * 65)
    print("       STAGE D: LIGHTWEIGHT EDGE INFERENCE SERVICE (ONNX)       ")
    print("=" * 65)
    print("[*] Initializing Edge Runtime Engine...")
    engine = EdgeSentimentClassifier()
    print("[*] ONNX Runtime session initialized with Graph Optimization: ENABLE_ALL")
    print("[*] Note: This script runs with ZERO PyTorch dependencies in memory!\n")

    # Warm-up (standard MLOps practice to avoid cold-start JIT bias)
    for _ in range(5):
        engine.predict("warmup engine test")

    test_queries = [
        "A truly remarkable storytelling and deep emotion in every scene",
        "This film was a catastrophic failure that should never have been made",
        "Delightful experience full of warmth and joy",
        "A cheap sloppy and completely forgettable mess",
        "Innovative direction and powerful emotional delivery"
    ]

    print("-" * 65)
    print(f"{'Input Sentence':<42} | {'Prediction':<9} | {'Conf':<6} | {'Latency'}")
    print("-" * 65)

    latencies = []
    for query in test_queries:
        res = engine.predict(query)
        latencies.append(res['latency_ms'])
        truncated_query = (query[:39] + "...") if len(query) > 42 else query
        print(f"{truncated_query:<42} | {res['sentiment']:<9} | {res['confidence']:.1f}% | {res['latency_ms']:.3f} ms")

    avg_latency = np.mean(latencies)
    print("-" * 65)
    print(f"[*] Edge Inference Average Latency: {avg_latency:.3f} ms per sample")
    print(f"[*] Theoretical Edge Throughput:    {1000.0 / avg_latency:.1f} inferences/sec")
    print("=" * 65)
    print("\n[SUCCESS] Stage D Complete! Edge deployment verified successfully.")

if __name__ == "__main__":
    main()
