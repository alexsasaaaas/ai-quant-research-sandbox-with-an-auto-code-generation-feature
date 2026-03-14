"""
io.py
I/O utilities for saving/loading data, models, and JSON summaries.
"""

import os
import json
import joblib
import yaml
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def save_json(data: Any, filepath: str):
    """Save data to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(data: Any, filepath: str):
    """Save data to YAML file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)


def load_yaml(filepath: str) -> Any:
    """Load data from YAML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_model(model: Any, filepath: str):
    """Serialize model using joblib."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(filepath: str) -> Any:
    """Load model using joblib."""
    return joblib.load(filepath)


def ensure_dir(path_str: str):
    """Ensure directory exists."""
    Path(path_str).mkdir(parents=True, exist_ok=True)
