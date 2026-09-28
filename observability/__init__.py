"""
Observability module for MedVision AI.

Provides Prometheus-based metrics for complete ViT pipeline monitoring:
- Model architecture introspection (parameters, layers, attention heads)
- Inference pipeline (latency, throughput, confidence distribution)
- Training pipeline (loss, accuracy, learning rate, gradient norms)
- Feature extraction & selection metrics
- Grad-CAM explainability metrics
- LLM report generation & evaluation metrics
- HTTP request/response metrics
"""
