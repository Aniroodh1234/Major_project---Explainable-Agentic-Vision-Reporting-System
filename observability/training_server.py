"""
Lightweight Prometheus HTTP server for batch training scripts.

Training agents (1-5) run as standalone Python processes — not inside
FastAPI — so they cannot share the ``/metrics`` endpoint.  This helper
starts a simple HTTP server on a configurable port so that Prometheus
can scrape live training metrics while the script is running.

Usage::

    from observability.training_server import start_training_metrics_server
    start_training_metrics_server(port=8001)
"""

import threading

from prometheus_client import start_http_server

from utils.logger import setup_logger

logger = setup_logger(__name__)

_server_started = False
_lock = threading.Lock()


def start_training_metrics_server(port: int = 8001) -> None:
    """
    Start a Prometheus-compatible HTTP metrics server.

    Safe to call multiple times — the server is started at most once
    per process.  If the port is already in use the error is logged
    but does **not** crash the training script.

    Args:
        port: TCP port for the HTTP server (default ``8001``).
    """
    global _server_started

    with _lock:
        if _server_started:
            logger.debug(f"Training metrics server already running on port {port}.")
            return
        try:
            start_http_server(port)
            _server_started = True
            logger.info(
                f"Prometheus training metrics server started -> "
                f"http://0.0.0.0:{port}/metrics"
            )
        except OSError as e:
            logger.warning(
                f"Could not start training metrics server on port {port}: {e}. "
                f"Training will continue without live metric scraping."
            )
