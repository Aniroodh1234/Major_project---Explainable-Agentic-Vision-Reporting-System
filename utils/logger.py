"""
Centralized logging configuration for the Agentic AI project.

Provides a reusable logger setup used by all agents and utility modules.
Uses Python's built-in logging module instead of print() statements.
"""

import logging
import sys
from pathlib import Path

from config.settings import LOG_DIR, LOG_LEVEL, LOG_FORMAT


def setup_logger(
    name: str,
    log_file: str | None = None,
) -> logging.Logger:
    """
    Create and configure a named logger instance.

    If the logger already has handlers attached (i.e. it was previously
    configured), the existing logger is returned without modification so
    that duplicate log entries are avoided.

    Args:
        name:     Name of the logger – typically ``__name__`` of the
                  calling module.
        log_file: Optional filename (not a full path).  When provided,
                  log output is *also* written to ``<LOG_DIR>/<log_file>``.

    Returns:
        A fully configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers on repeated calls.
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(LOG_FORMAT)

    # ---- Console handler (stdout) ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ---- Optional file handler ----
    if log_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            LOG_DIR / log_file,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def print_step(step_num: int, title: str, description: str, color: str = "\033[1;36m") -> None:
    """
    Prints a beautiful, layman-readable narrative step to the terminal.
    
    Args:
        step_num: The step number in the pipeline.
        title: Short title of the action.
        description: Layman explanation of what is happening.
        color: ANSI color code (default cyan).
    """
    reset = "\033[0m"
    bold = "\033[1m"
    green = "\033[1;32m"
    
    print(f"\n{color}{bold}============================================================{reset}")
    print(f"{color}{bold} STEP {step_num}: {title}{reset}")
    print(f"{color}{bold}============================================================{reset}")
    print(f"{green}>> {description}{reset}")
    
def print_substep(description: str) -> None:
    """Prints a beautiful sub-step."""
    yellow = "\033[1;33m"
    reset = "\033[0m"
    print(f"  {yellow}• {description}{reset}")

