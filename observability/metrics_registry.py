"""
Centralised Prometheus metrics registry for MedVision AI.

Every metric used across the project is defined here exactly once.
All modules import from this file to guarantee consistency and avoid
duplicate metric registration errors.
"""

from prometheus_client import Counter, Gauge, Histogram, Info


# ══════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE & STATE
# ══════════════════════════════════════════════════════════════════════

MODEL_INFO = Info(
    "medvision_model",
    "Static information about the loaded Vision Transformer model",
)

MODEL_LOADED = Gauge(
    "medvision_model_loaded",
    "Whether the ViT classifier is loaded and ready (1 = yes, 0 = no)",
)

MODEL_PARAMETERS = Gauge(
    "medvision_model_parameters",
    "Number of parameters in the model",
    ["component", "trainable"],
)

MODEL_ENCODER_LAYER_PARAMS = Gauge(
    "medvision_vit_encoder_layer_parameters",
    "Number of parameters per ViT encoder layer",
    ["layer_index"],
)


# ══════════════════════════════════════════════════════════════════════
#  INFERENCE PIPELINE  (Agent 6)
# ══════════════════════════════════════════════════════════════════════

INFERENCE_REQUESTS_TOTAL = Counter(
    "medvision_inference_requests_total",
    "Total number of inference requests processed",
    ["status"],
)

INFERENCE_DURATION_SECONDS = Histogram(
    "medvision_inference_duration_seconds",
    "Total end-to-end inference time (preprocessing + forward + GradCAM)",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

PREPROCESSING_DURATION_SECONDS = Histogram(
    "medvision_preprocessing_duration_seconds",
    "Time to preprocess a single image (resize, normalize, tensorize)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

FORWARD_PASS_DURATION_SECONDS = Histogram(
    "medvision_forward_pass_duration_seconds",
    "ViT model forward-pass latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

SOFTMAX_ENTROPY = Histogram(
    "medvision_softmax_entropy",
    "Entropy of the softmax output distribution (lower = more decisive)",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
)

PREDICTIONS_TOTAL = Counter(
    "medvision_predictions_total",
    "Total predictions broken down by predicted class",
    ["predicted_class"],
)

CONFIDENCE_SCORE = Histogram(
    "medvision_confidence_score",
    "Distribution of prediction confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
)

CONFIDENCE_LATEST = Gauge(
    "medvision_confidence_score_latest",
    "Most recent prediction confidence score",
)

LOW_CONFIDENCE_TOTAL = Counter(
    "medvision_low_confidence_predictions_total",
    "Total predictions that fell below the confidence threshold",
)

INPUT_IMAGE_SIZE = Histogram(
    "medvision_input_image_pixels",
    "Dimension of the uploaded input image in pixels",
    ["dimension"],
    buckets=[64, 128, 224, 256, 512, 1024, 2048, 4096],
)


# ══════════════════════════════════════════════════════════════════════
#  GRAD-CAM  /  EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════

GRADCAM_DURATION_SECONDS = Histogram(
    "medvision_gradcam_duration_seconds",
    "Time to generate a Grad-CAM heatmap",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

GRADCAM_ACTIVATION_MEAN = Histogram(
    "medvision_gradcam_activation_mean",
    "Mean intensity of the Grad-CAM activation map (0-1 scale)",
    buckets=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7],
)

GRADCAM_ACTIVATION_MAX = Gauge(
    "medvision_gradcam_activation_max",
    "Maximum activation value in the latest Grad-CAM heatmap",
)

GRADCAM_ACTIVE_REGION_RATIO = Histogram(
    "medvision_gradcam_active_region_ratio",
    "Fraction of heatmap pixels with activation > 0.5 (focus area)",
    buckets=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7],
)

GRADCAM_SPATIAL_CONCENTRATION = Histogram(
    "medvision_gradcam_spatial_concentration",
    "Std-dev of activation values (lower = more spatially focused)",
    buckets=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
)


# ══════════════════════════════════════════════════════════════════════
#  LLM  /  REPORT GENERATION  (Agent 7)
# ══════════════════════════════════════════════════════════════════════

LLM_INVOCATIONS_TOTAL = Counter(
    "medvision_llm_invocations_total",
    "Total LLM API invocations",
    ["purpose"],
)

LLM_DURATION_SECONDS = Histogram(
    "medvision_llm_duration_seconds",
    "LLM API call latency",
    ["purpose"],
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

LLM_TOKENS_TOTAL = Counter(
    "medvision_llm_tokens_total",
    "Total LLM tokens consumed",
    ["type"],
)

LLM_ERRORS_TOTAL = Counter(
    "medvision_llm_errors_total",
    "Total LLM invocation errors (retries counted separately)",
)

REPORT_GENERATION_DURATION_SECONDS = Histogram(
    "medvision_report_generation_duration_seconds",
    "Full report generation + evaluation pipeline time",
    buckets=[1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0],
)


# ══════════════════════════════════════════════════════════════════════
#  AGENT 8 — EVALUATION  /  JUDGE
# ══════════════════════════════════════════════════════════════════════

EVALUATION_SCORE_PERCENT = Histogram(
    "medvision_evaluation_score_percent",
    "Report quality evaluation score (0-100 scale)",
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100],
)

EVALUATION_SCORE_LATEST = Gauge(
    "medvision_evaluation_score_latest",
    "Most recent evaluation score percentage",
)

EVALUATION_ITERATIONS = Histogram(
    "medvision_evaluation_iterations_count",
    "Number of refinement iterations needed per report",
    buckets=[1, 2, 3, 4, 5],
)

EVALUATION_CRITERION_SCORE = Gauge(
    "medvision_evaluation_criterion_score",
    "Score per individual evaluation criterion (0/1/2)",
    ["criterion_id"],
)

EVALUATION_PASSED_TOTAL = Counter(
    "medvision_evaluation_passed_total",
    "Total reports that passed the quality threshold",
)

EVALUATION_FAILED_TOTAL = Counter(
    "medvision_evaluation_failed_total",
    "Total reports that failed the quality threshold (partially validated)",
)


# ══════════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE  (Agent 5)
# ══════════════════════════════════════════════════════════════════════

TRAINING_EPOCH_CURRENT = Gauge(
    "medvision_training_epoch_current",
    "Currently executing training epoch number",
)

TRAINING_EPOCH_TOTAL = Gauge(
    "medvision_training_epochs_total",
    "Total number of training epochs configured",
)

TRAINING_LOSS = Gauge(
    "medvision_training_loss",
    "Loss value at the end of each epoch",
    ["phase"],
)

TRAINING_ACCURACY = Gauge(
    "medvision_training_accuracy",
    "Accuracy value at the end of each epoch",
    ["phase"],
)

TRAINING_LEARNING_RATE = Gauge(
    "medvision_training_learning_rate",
    "Current optimizer learning rate",
)

TRAINING_BEST_VAL_ACCURACY = Gauge(
    "medvision_training_best_validation_accuracy",
    "Best validation accuracy achieved so far during training",
)

TRAINING_EPOCH_DURATION_SECONDS = Histogram(
    "medvision_training_epoch_duration_seconds",
    "Wall-clock duration of each training epoch",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

TRAINING_BATCH_DURATION_SECONDS = Histogram(
    "medvision_training_batch_duration_seconds",
    "Duration of each mini-batch forward + backward pass",
    ["phase"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

TRAINING_GRADIENT_NORM = Gauge(
    "medvision_training_gradient_norm",
    "L2 norm of gradients per model component",
    ["component"],
)

TRAINING_DATASET_SIZE = Gauge(
    "medvision_training_dataset_size",
    "Number of samples in each dataset split",
    ["split"],
)

TRAINING_EARLY_STOPPING_COUNTER = Gauge(
    "medvision_training_early_stopping_counter",
    "Number of consecutive epochs without validation improvement",
)


# ══════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (Agent 3)
# ══════════════════════════════════════════════════════════════════════

FEATURES_EXTRACTED_TOTAL = Counter(
    "medvision_features_extracted_total",
    "Total feature embeddings extracted",
    ["class_name"],
)

FEATURE_EXTRACTION_DURATION_SECONDS = Histogram(
    "medvision_feature_extraction_duration_seconds",
    "Time per single feature extraction operation",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

FEATURE_VECTOR_L2_NORM = Histogram(
    "medvision_feature_vector_l2_norm",
    "L2 norm of extracted 768-dim feature vectors",
    buckets=[1, 5, 10, 15, 20, 30, 50, 100],
)

FEATURE_VECTOR_MEAN = Histogram(
    "medvision_feature_vector_mean",
    "Mean value of the extracted feature vector components",
    buckets=[-1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 5],
)

FEATURE_VECTOR_SPARSITY = Histogram(
    "medvision_feature_vector_sparsity",
    "Fraction of near-zero elements in the feature vector (abs < 0.01)",
    buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# ══════════════════════════════════════════════════════════════════════
#  DATA PIPELINE  (Agents 1-2)
# ══════════════════════════════════════════════════════════════════════

DATASET_IMAGES_TOTAL = Gauge(
    "medvision_dataset_images_total",
    "Total images at each pipeline stage",
    ["stage", "class_name"],
)

DATASET_CORRUPTED_TOTAL = Gauge(
    "medvision_dataset_corrupted_total",
    "Total corrupted images detected during cleaning",
)

DATASET_DUPLICATES_REMOVED_TOTAL = Gauge(
    "medvision_dataset_duplicates_removed_total",
    "Total duplicate images removed during cleaning",
)

PREPROCESSING_IMAGES_TOTAL = Counter(
    "medvision_preprocessing_images_total",
    "Total images that went through the preprocessing pipeline",
    ["status"],
)

PREPROCESSING_THROUGHPUT = Gauge(
    "medvision_preprocessing_throughput_images_per_second",
    "Current preprocessing throughput (images per second)",
)


# ══════════════════════════════════════════════════════════════════════
#  FEATURE SELECTION / VALIDATION  (Agent 4)
# ══════════════════════════════════════════════════════════════════════

FEATURE_VALIDATION_TOTAL = Counter(
    "medvision_feature_validation_total",
    "Total feature embeddings validated",
    ["result"],
)


# ══════════════════════════════════════════════════════════════════════
#  HTTP / API  (FastAPI middleware)
# ══════════════════════════════════════════════════════════════════════

HTTP_REQUESTS_TOTAL = Counter(
    "medvision_http_requests_total",
    "Total HTTP requests to the FastAPI backend",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "medvision_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "medvision_http_requests_in_progress",
    "HTTP requests currently being processed",
    ["method", "endpoint"],
)
