import json
import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# -------------------------------------------------------------
# Cloud-Style REST API Service (FastAPI + ONNX Runtime)
# -------------------------------------------------------------
# In a modern MLOps production pipeline, models are served via
# asynchronous microservice APIs (FastAPI) wrapped in Docker containers.
# -------------------------------------------------------------

app = FastAPI(
    title="High-Performance Sentiment Inference API",
    description="Production-grade REST microservice for low-latency sentiment inference powered by ONNX Runtime.",
    version="1.0.0"
)

# Global model state
session = None
vocab = None
input_name = None
output_name = None
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}
MAX_SEQ_LEN = 16

@app.on_event("startup")
def load_artifacts():
    global session, vocab, input_name, output_name
    print("[*] Loading ONNX model and vocabulary into memory...")
    with open("artifacts/vocab.json", "r") as f:
        vocab = json.load(f)

    session_opts = ort.SessionOptions()
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession("artifacts/model.onnx", session_opts, providers=["CPUExecutionProvider"])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("[*] REST API microservice ready for incoming HTTP inference traffic.")

def preprocess_text(text: str) -> List[int]:
    tokens = text.lower().split()
    indices = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    if len(indices) < MAX_SEQ_LEN:
        indices += [vocab["<PAD>"]] * (MAX_SEQ_LEN - len(indices))
    else:
        indices = indices[:MAX_SEQ_LEN]
    return indices

# Request & Response Schemas
class PredictRequest(BaseModel):
    text: str = Field(..., example="This movie was an absolute masterpiece of cinema!")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., example=[
        "Fantastic acting and brilliant direction!",
        "A catastrophic and boring waste of time."
    ])

class PredictionItem(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict

class PredictResponse(BaseModel):
    prediction: PredictionItem
    server_latency_ms: float

class BatchPredictResponse(BaseModel):
    predictions: List[PredictionItem]
    batch_size: int
    server_latency_ms: float
    throughput_samples_per_sec: float

@app.get("/", tags=["General"])
def root():
    return {
        "service": "High-Performance Sentiment Inference API",
        "runtime": "ONNX Runtime (CPU)",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health", tags=["Monitoring"])
def health_check():
    # In cloud environments (Kubernetes, AWS ECS), health checks determine
    # whether traffic should be routed to this container instance.
    if session is None or vocab is None:
        raise HTTPException(status_code=503, detail="Model artifacts not yet loaded.")
    return {
        "status": "HEALTHY",
        "model_format": "ONNX (v1.14)",
        "memory_status": "Ready for inference"
    }

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_single(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    t0 = time.perf_counter()
    indices = preprocess_text(req.text)
    input_arr = np.array([indices], dtype=np.int64)

    outputs = session.run([output_name], {input_name: input_arr})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    pred_class = int(np.argmax(probs))

    return PredictResponse(
        prediction=PredictionItem(
            text=req.text,
            sentiment=LABELS[pred_class],
            confidence=round(float(probs[pred_class] * 100.0), 2),
            probabilities={
                "NEGATIVE": round(float(probs[0] * 100.0), 2),
                "POSITIVE": round(float(probs[1] * 100.0), 2)
            }
        ),
        server_latency_ms=round(latency_ms, 3)
    )

@app.post("/predict-batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(req: BatchPredictRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="Batch list cannot be empty.")

    t0 = time.perf_counter()
    batch_indices = [preprocess_text(t) for t in req.texts]
    input_arr = np.array(batch_indices, dtype=np.int64)

    outputs = session.run([output_name], {input_name: input_arr})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    predictions = []
    logits_matrix = outputs[0]
    for idx, text in enumerate(req.texts):
        logits = logits_matrix[idx]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        pred_class = int(np.argmax(probs))

        predictions.append(
            PredictionItem(
                text=text,
                sentiment=LABELS[pred_class],
                confidence=round(float(probs[pred_class] * 100.0), 2),
                probabilities={
                    "NEGATIVE": round(float(probs[0] * 100.0), 2),
                    "POSITIVE": round(float(probs[1] * 100.0), 2)
                }
            )
        )

    throughput = (len(req.texts) / latency_ms) * 1000.0

    return BatchPredictResponse(
        predictions=predictions,
        batch_size=len(req.texts),
        server_latency_ms=round(latency_ms, 3),
        throughput_samples_per_sec=round(throughput, 1)
    )
