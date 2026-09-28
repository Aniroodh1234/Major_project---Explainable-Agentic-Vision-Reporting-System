"""
Vision Transformer architecture introspection.

Walks the MedicalClassifierViT model graph and records structural
metrics (parameter counts, layer dimensions, attention head config)
into Prometheus gauges so they are always visible on the dashboard.
"""

from typing import Any, Dict

import torch.nn as nn

from observability.metrics_registry import (
    MODEL_ENCODER_LAYER_PARAMS,
    MODEL_INFO,
    MODEL_PARAMETERS,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def inspect_vit_model(model: nn.Module) -> Dict[str, Any]:
    """
    Introspect a ``MedicalClassifierViT`` model and publish architecture
    metrics to the Prometheus default registry.

    Metrics recorded
    ----------------
    * ``medvision_model`` (Info) – static key/value pairs describing the
      model (backbone, num_classes, hidden_dim, …).
    * ``medvision_model_parameters`` (Gauge) – parameter counts per
      component (backbone, encoder, classifier, total) and trainability.
    * ``medvision_vit_encoder_layer_parameters`` (Gauge) – parameter
      count per individual encoder block.

    Args:
        model: An instance of ``MedicalClassifierViT``.

    Returns:
        Dictionary of architecture details (also logged to console).
    """
    arch: Dict[str, Any] = {}

    # ── Total parameter counts ──────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    MODEL_PARAMETERS.labels(component="total", trainable="all").set(total_params)
    MODEL_PARAMETERS.labels(component="total", trainable="yes").set(trainable_params)
    MODEL_PARAMETERS.labels(component="total", trainable="no").set(frozen_params)

    arch["total_parameters"] = total_params
    arch["trainable_parameters"] = trainable_params
    arch["frozen_parameters"] = frozen_params

    # ── Feature-extractor backbone ──────────────────────────────────
    if hasattr(model, "feature_extractor") and hasattr(model.feature_extractor, "backbone"):
        vit = model.feature_extractor.backbone
        backbone_params = sum(p.numel() for p in vit.parameters())
        MODEL_PARAMETERS.labels(component="backbone", trainable="all").set(backbone_params)
        arch["backbone_parameters"] = backbone_params

        # Patch embedding (conv_proj in torchvision ViT)
        if hasattr(vit, "conv_proj"):
            patch_params = sum(p.numel() for p in vit.conv_proj.parameters())
            MODEL_PARAMETERS.labels(component="patch_embedding", trainable="all").set(patch_params)
            arch["patch_embedding_parameters"] = patch_params

            # Kernel size = patch size
            if hasattr(vit.conv_proj, "kernel_size"):
                arch["patch_size"] = str(vit.conv_proj.kernel_size)

        # Encoder layers
        if hasattr(vit, "encoder") and hasattr(vit.encoder, "layers"):
            encoder_layers = vit.encoder.layers
            num_layers = len(encoder_layers)
            arch["num_encoder_layers"] = num_layers

            for idx, layer in enumerate(encoder_layers):
                layer_params = sum(p.numel() for p in layer.parameters())
                MODEL_ENCODER_LAYER_PARAMS.labels(layer_index=str(idx)).set(layer_params)

                # Extract attention-head info from the first layer
                if idx == 0 and hasattr(layer, "self_attention"):
                    attn = layer.self_attention
                    if hasattr(attn, "num_heads"):
                        arch["num_attention_heads"] = attn.num_heads
                    if hasattr(attn, "head_dim"):
                        arch["head_dim"] = attn.head_dim
                    # Hidden dim from in_proj_weight shape
                    if hasattr(attn, "in_proj_weight") and attn.in_proj_weight is not None:
                        arch["hidden_dimension"] = attn.in_proj_weight.shape[1]

            encoder_params = sum(p.numel() for p in vit.encoder.parameters())
            MODEL_PARAMETERS.labels(component="encoder", trainable="all").set(encoder_params)
            arch["encoder_parameters"] = encoder_params

    # ── Classification head ─────────────────────────────────────────
    if hasattr(model, "classifier"):
        clf_params = sum(p.numel() for p in model.classifier.parameters())
        MODEL_PARAMETERS.labels(component="classifier_head", trainable="all").set(clf_params)
        arch["classifier_head_parameters"] = clf_params
        arch["num_classes"] = getattr(model, "num_classes", "unknown")

    # ── Publish static Info metric ──────────────────────────────────
    info_labels = {k: str(v) for k, v in arch.items()}
    MODEL_INFO.info(info_labels)

    logger.info(
        f"ViT Model Inspection: {total_params:,} total params "
        f"({trainable_params:,} trainable, {frozen_params:,} frozen), "
        f"{arch.get('num_encoder_layers', '?')} encoder layers, "
        f"{arch.get('num_attention_heads', '?')} attention heads"
    )

    return arch
