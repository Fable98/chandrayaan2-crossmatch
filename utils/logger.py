"""
utils/logger.py — Centralized Structured Logging Configuration

Provides unified logging across all pipeline components in compliance with:
%(asctime)s - %(name)s - %(levelname)s - %(message)s
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
    format_str: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """
    Configures the root and package loggers.

    Args:
        log_file: Optional path to an output log file (e.g., pipeline.log).
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        format_str: Log message format string.
        date_format: Date format string for timestamps.

    Returns:
        The configured root logger.
    """
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Check if file handler already exists for this path
        existing_paths = [
            getattr(h, "baseFilename", None)
            for h in root_logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        if str(log_path.resolve()) not in existing_paths:
            file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance with the given module name.
    """
    return logging.getLogger(name)
