from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any


# 참고 로그(testlog) 양식: 2026-07-20 07:17:17,036 [INFO] message
# datefmt 를 지정하지 않으면 asctime 뒤에 밀리초(,mmm)가 자동 포함된다.
_LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
_BANNER_WIDTH = 60


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

    formatter = logging.Formatter(fmt=_LOG_FMT)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.log_path = log_path  # type: ignore[attr-defined]
    logger.info("Log file: %s", log_path)
    return logger


def save_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def log_banner(
    logger: logging.Logger,
    title: str,
    char: str = "=",
    width: int = _BANNER_WIDTH,
) -> None:
    """Centered section banner, e.g. ==== Configurations ====."""
    pad = max(0, width - len(title) - 2)
    left = pad // 2
    right = pad - left
    logger.info("%s %s %s", char * left, title, char * right)


def log_configurations(logger: logging.Logger, config: dict[str, Any]) -> None:
    banner = _centered_banner("Configurations", char="=")
    body = json.dumps(config, indent=2, ensure_ascii=False, default=str)
    logger.info("%s\n%s", banner, body)


def log_evaluation_summary(
    logger: logging.Logger,
    primary: dict[str, Any],
    primary_label: str,
    primary_note: str,
    reference: dict[str, Any] | None = None,
    reference_label: str | None = None,
    reference_note: str | None = None,
    ranking: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    subtitle: str = "(포인트 보정 후 — all classification metrics)",
) -> None:
    """
    Boxed evaluation summary matching the testlog style.

    Example:
      ============================================================
       EVALUATION SUMMARY
       (포인트 보정 후 — all classification metrics)
      ============================================================
      [validation]  포인트 보정 후 | primary (threshold from validation)
      ------------------------------------------------------------
        F1          0.5000
        ...
    """
    lines: list[str] = [
        "",
        "=" * _BANNER_WIDTH,
        " EVALUATION SUMMARY",
        f" {subtitle}",
        "=" * _BANNER_WIDTH,
        "",
    ]

    lines.append(_section_header(primary_label, primary_note))
    lines.append("-" * _BANNER_WIDTH)
    lines.extend(_format_kv_lines(primary))

    if reference is not None and reference_label is not None:
        lines.append("")
        lines.append(_section_header(reference_label, reference_note or ""))
        lines.append("-" * _BANNER_WIDTH)
        lines.extend(_format_kv_lines(reference))

    if ranking is not None:
        lines.append("")
        lines.append(
            "[Ranking]  포인트 보정 후 | AUROC / AUPRC (PA at each threshold)"
        )
        lines.append("-" * _BANNER_WIDTH)
        lines.extend(_format_kv_lines(ranking))

    if training:
        lines.append("")
        lines.append("[Training]")
        lines.append("-" * _BANNER_WIDTH)
        lines.extend(_format_kv_lines(training))

    lines.append("=" * _BANNER_WIDTH)
    logger.info("\n".join(lines))


def _section_header(label: str, note: str) -> str:
    return f"[{label}]  포인트 보정 후 | {note}"


def _centered_banner(title: str, char: str = "=", width: int = _BANNER_WIDTH) -> str:
    pad = max(0, width - len(title) - 2)
    left = pad // 2
    right = pad - left
    return f"{char * left} {title} {char * right}"


def _format_kv_lines(
    items: dict[str, Any] | list[tuple[str, Any]],
    key_width: int = 16,
) -> list[str]:
    pairs = items.items() if isinstance(items, dict) else items
    lines: list[str] = []
    for key, value in pairs:
        if key == "Threshold" and isinstance(value, (float, int)):
            lines.append(f"  {key:<{key_width}} {float(value):.6f}")
        elif isinstance(value, float):
            lines.append(f"  {key:<{key_width}} {value:.4f}")
        else:
            lines.append(f"  {key:<{key_width}} {value}")
    return lines
