# tests/test_logger.py
"""Tests for the logging configuration."""

import logging
import tempfile
from pathlib import Path

from src.logger import configure_logging


def test_configure_logging_creates_log_file():
    """Logger should create the log file on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.log"
        configure_logging(log_level="INFO", log_file=log_path)
        logger = logging.getLogger("test_logger")
        logger.info("test message")
        assert log_path.exists()


def test_configure_logging_creates_parent_dirs():
    """Logger should create parent directories if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "subdir" / "deep" / "test.log"
        configure_logging(log_level="DEBUG", log_file=log_path)
        logger = logging.getLogger("test_deep")
        logger.debug("deep message")
        assert log_path.exists()
