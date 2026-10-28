import os
import json
import pytest
import numpy as np

def test_vocab_exists():
    """Verify that the tokenizer vocabulary artifact was generated and is valid."""
    vocab_path = "artifacts/vocab.json"
    assert os.path.exists(vocab_path), "artifacts/vocab.json does not exist!"
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    assert "<PAD>" in vocab, "<PAD> token missing from vocabulary!"
    assert "<UNK>" in vocab, "<UNK> token missing from vocabulary!"
    assert len(vocab) > 10, "Vocabulary has fewer than 10 tokens!"

def test_serialized_artifacts_exist():
    """Verify that all three serialized model formats exist."""
    assert os.path.exists("artifacts/model.pkl"), "artifacts/model.pkl missing!"
    assert os.path.exists("artifacts/model.pt"), "artifacts/model.pt missing!"
    assert os.path.exists("artifacts/model.onnx"), "artifacts/model.onnx missing!"

def test_edge_inference():
    """Verify that the ONNX Runtime edge inference engine produces valid predictions."""
    from edge_infer import EdgeSentimentClassifier
    classifier = EdgeSentimentClassifier()
    result = classifier.predict("This movie was wonderful and brilliant!")
    assert result["sentiment"] in ["POSITIVE", "NEGATIVE"]
    assert 0.0 <= result["confidence"] <= 100.0
    assert result["latency_ms"] > 0

def test_fastapi_health_endpoint():
    """Verify that the FastAPI cloud microservice health check returns 200 HEALTHY."""
    import app
    app.load_artifacts()
    response = app.health_check()
    assert response["status"] == "HEALTHY"
    assert response["model_format"] == "ONNX (v1.14)"
