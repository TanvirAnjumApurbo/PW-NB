"""Utility functions: logging, seeding, helpers."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger for the project."""
    logger = logging.getLogger("pwnb")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def seed_everything(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT
