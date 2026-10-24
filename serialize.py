import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

# -------------------------------------------------------------
# 1. Model Definition (Identical to Training)
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

# -------------------------------------------------------------
# 2. Main Serialization Pipeline
# -------------------------------------------------------------
def main():
    print("=" * 65)
    print("       STAGE C: MODEL SERIALIZATION & PARITY VERIFICATION       ")
    print("=" * 65)

    # Load Vocabulary
    with open("artifacts/vocab.json", "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"[*] Loaded vocabulary ({vocab_size} tokens).")

    # Reconstruct Trained Model
    model = SentimentClassifier(vocab_size=vocab_size)
    weights_path = "artifacts/model_weights.pt"
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    print(f"[*] Loaded trained PyTorch weights from: {weights_path}")

    # ---------------------------------------------------------
    # FORMAT 1: Pickle (.pkl)
    # ---------------------------------------------------------
    pickle_path = "artifacts/model.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(model, f)
    pickle_size = os.path.getsize(pickle_path) / 1024
    print(f"\n[1] Pickle Serialization (.pkl):")
    print(f"    - File: {pickle_path} ({pickle_size:.2f} KB)")
    print(f"    - Info: Python object serialization. Requires Python & exact class definitions.")
    print(f"    - Security Note: Load ONLY trusted .pkl files (arbitrary code execution risk).")

    # ---------------------------------------------------------
    # FORMAT 2: Native PyTorch Checkpoint (.pt)
    # ---------------------------------------------------------
    pt_path = "artifacts/model.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "vocab_size": vocab_size,
            "embed_dim": 64,
            "hidden_dim": 32,
            "num_classes": 2,
            "max_seq_length": 16,
        }
    }
    torch.save(checkpoint, pt_path)
    pt_size = os.path.getsize(pt_path) / 1024
    print(f"\n[2] Native PyTorch Serialization (.pt):")
    print(f"    - File: {pt_path} ({pt_size:.2f} KB)")
    print(f"    - Info: Serializes pure tensor weights and architecture metadata.")

    # ---------------------------------------------------------
    # FORMAT 3: ONNX (.onnx)
    # ---------------------------------------------------------
    onnx_path = "artifacts/model.onnx"
    dummy_input = torch.zeros((1, 16), dtype=torch.long)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        dynamo=False
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_size = os.path.getsize(onnx_path) / 1024
    print(f"\n[3] Open Neural Network Exchange (.onnx):")
    print(f"    - File: {onnx_path} ({onnx_size:.2f} KB)")
    print(f"    - Info: Open-standard graph format with dynamic batching. Portable to C++, iOS, Edge.")

    # ---------------------------------------------------------
    # 3. Parity Verification (Comparing All Three Runtimes)
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("      NUMERICAL PARITY CHECK ACROSS ALL 3 SERIALIZED FORMATS   ")
    print("=" * 65)

    # 1. Load Pickle
    with open(pickle_path, "rb") as f:
        pickle_loaded_model = pickle.load(f)
        pickle_loaded_model.eval()

    # 2. Load Native PyTorch
    pt_loaded_ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    pt_loaded_model = SentimentClassifier(vocab_size=pt_loaded_ckpt["metadata"]["vocab_size"])
    pt_loaded_model.load_state_dict(pt_loaded_ckpt["model_state_dict"])
    pt_loaded_model.eval()

    # 3. Load ONNX Runtime
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    test_sentences = [
        "an absolute masterpiece of cinema",
        "a cheap sloppy and completely forgettable mess",
        "wonderfully written and expertly directed",
        "horrible acting and dreadful special effects"
    ]

    def tokenize(text):
        tokens = text.lower().split()
        indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
        if len(indices) < 16:
            indices += [vocab["<PAD>"]] * (16 - len(indices))
        else:
            indices = indices[:16]
        return indices

    batch_indices = [tokenize(s) for s in test_sentences]
    torch_input = torch.tensor(batch_indices, dtype=torch.long)
    numpy_input = np.array(batch_indices, dtype=np.int64)

    # Run Inferences
    with torch.no_grad():
        pickle_logits = pickle_loaded_model(torch_input).numpy()
        pt_logits = pt_loaded_model(torch_input).numpy()

    onnx_outputs = session.run(["logits"], {"input_ids": numpy_input})
    onnx_logits = onnx_outputs[0]

    # Verify Numerical Equivalence
    diff_pt_pickle = np.max(np.abs(pt_logits - pickle_logits))
    diff_pt_onnx = np.max(np.abs(pt_logits - onnx_logits))

    print(f"\n[*] Maximum Absolute Discrepancy:")
    print(f"    - |PyTorch - Pickle|: {diff_pt_pickle:.8f}")
    print(f"    - |PyTorch - ONNX|:   {diff_pt_onnx:.8f}")

    if diff_pt_pickle < 1e-5 and diff_pt_onnx < 1e-4:
        print("    -> [VERIFIED] Perfect Mathematical Consistency Across All Runtimes!")
    else:
        print("    -> [WARNING] Tolerance threshold exceeded.")

    print("\n[*] Sample Sentence Predictions:")
    labels_map = {0: "NEGATIVE", 1: "POSITIVE"}
    for idx, sentence in enumerate(test_sentences):
        pred_class = int(np.argmax(onnx_logits[idx]))
        probs = np.exp(onnx_logits[idx]) / np.sum(np.exp(onnx_logits[idx]))
        confidence = probs[pred_class] * 100.0
        print(f"    Sentence: \"{sentence}\"")
        print(f"    Prediction: {labels_map[pred_class]} ({confidence:.1f}% confidence)\n")

    print("[SUCCESS] Stage C Complete! All 3 artifacts (.pkl, .pt, .onnx) ready for Edge & Cloud deployment.")

if __name__ == "__main__":
    main()
