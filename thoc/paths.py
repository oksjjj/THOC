"""Output path layout aligned with OmniAnomaly: ``{root}/{dataset}/{exp}/``."""

from __future__ import annotations

import os


def resolve_output_dirs(
    dataset: str,
    exp_name: str,
    save_root: str = "model",
    result_root: str = "result",
    log_root: str = "log",
) -> tuple[str, str, str]:
    """
    Nest outputs under ``{root}/{dataset}/{exp_name}/``.

    Returns:
        (save_dir, result_dir, log_dir) — full experiment directories.
    """
    save_dir = os.path.join(save_root, dataset, exp_name)
    result_dir = os.path.join(result_root, dataset, exp_name)
    log_dir = os.path.join(log_root, dataset, exp_name)
    return save_dir, result_dir, log_dir
