from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any


def setup_logger(
    name: str,
    log_dir: str,
    exp_name: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger that writes to console and a log file."""
    os.makedirs(log_dir, exist_ok=True)
    exp_name = exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{exp_name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.log_path = log_path  # type: ignore[attr-defined]
    return logger


def save_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
