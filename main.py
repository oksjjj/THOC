#!/usr/bin/env python3
"""
THOC 학습/평가 진입점.

실행 흐름:
  1. 데이터 로드 및 논문 프로토콜 train/val/test 분할
  2. THOC 모델 학습 — validation F1-PA 기준 checkpoint
  3. validation threshold 로 test 평가 (F1, F1-PA, AUC)
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch

from thoc.data import (
    build_dataloaders,
    get_dataset_defaults,
    load_dataset,
    make_dataset_splits,
)
from thoc.logger import save_json, setup_logger
from thoc.model import THOC
from thoc.trainer import THOCTrainer, TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_exp_name(
    dataset: str,
    window_size: int,
    epochs: int,
    suffix: str = "",
) -> str:
    parts = [dataset.lower(), f"ws{window_size}", f"ep{epochs}"]
    if suffix:
        parts.append(suffix)
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return "_".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="THOC: Temporal Hierarchical One-Class Network"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="NeurIPS-TS-MUL",
        choices=["NeurIPS-TS-UNI", "NeurIPS-TS-MUL", "SMAP", "SMD"],
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--test_stride", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument(
        "--no_val_split",
        action="store_true",
        help="논문 train/val/test 분할 비활성화 (레거시)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=192)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=1.0)
    parser.add_argument("--lambda_orth", type=float, default=1.0)
    parser.add_argument("--lambda_tss", type=float, default=1.0)
    parser.add_argument("--scaler", type=str, default="std", choices=["std", "minmax"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--infer_threshold_policy",
        type=str,
        default="validation",
        choices=["validation", "test_best_f1_pa"],
        help="test 평가 threshold: validation(기본, 논문) 또는 test_best_f1_pa(참고)",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="학습 생략, --checkpoint 로 지정한 가중치로 test infer 만 실행",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="eval_only 시 사용할 .pt 경로 (미지정 시 checkpoints/{exp_name}/best.pt)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="lr, lambda_orth, lambda_tss 그리드 서치 후 최적 조합으로 재학습",
    )
    parser.add_argument(
        "--tune_epochs",
        type=int,
        default=15,
        help="--tune 시 각 조합당 학습 epoch (기본 15)",
    )
    return parser.parse_args()


TUNE_GRID: dict[str, list[float]] = {
    "lr": [1e-4, 5e-4, 1e-3],
    "lambda_orth": [0.01, 0.1, 1.0],
    "lambda_tss": [0.01, 0.1, 0.5, 1.0],
}


def run_training(
    args: argparse.Namespace,
    logger: logging.Logger,
    device: torch.device,
    lr: float,
    lambda_orth: float,
    lambda_tss: float,
    epochs: int,
    exp_name: str,
) -> dict[str, float]:
    defaults = get_dataset_defaults(args.dataset)
    data_dir = args.data_dir or defaults.data_dir
    window_size = args.window_size or defaults.window_size
    stride = args.stride or defaults.stride
    test_stride = args.test_stride if args.test_stride is not None else defaults.test_stride

    exp_checkpoint_dir = os.path.join(args.checkpoint_dir, exp_name)
    exp_output_dir = os.path.join(args.output_dir, exp_name)

    train_x, train_y, test_x, test_y = load_dataset(args.dataset, data_dir)

    if args.no_val_split:
        splits = make_dataset_splits(
            args.dataset, train_x, train_y, test_x, test_y, val_ratio=0.0
        )
        val_x, val_y = None, None
    else:
        splits = make_dataset_splits(
            args.dataset, train_x, train_y, test_x, test_y, val_ratio=args.val_ratio
        )
        val_x, val_y = splits.val_x, splits.val_y

    logger.info(
        "Splits | train: %s | val: %s | test: %s",
        splits.train_x.shape,
        None if val_x is None else val_x.shape,
        splits.test_x.shape,
    )
    if val_y is not None:
        logger.info(
            "Split anomaly ratio | val: %.2f%% | test: %.2f%%",
            100.0 * val_y.mean(),
            100.0 * splits.test_y.mean(),
        )

    _, train_loader, _, val_loader, _, test_loader = build_dataloaders(
        splits.train_x,
        splits.train_y,
        splits.test_x,
        splits.test_y,
        window_size=window_size,
        stride=stride,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        scaler_name=args.scaler,
        test_stride=test_stride,
        val_x=val_x,
        val_y=val_y,
    )
    logger.info(
        "Dataloaders | train windows: %d | val windows: %s | test windows: %d",
        len(train_loader.dataset),
        len(val_loader.dataset) if val_loader else "N/A",
        len(test_loader.dataset),
    )

    n_channels = splits.train_x.shape[1]
    model = THOC(
        n_channels=n_channels,
        window_size=window_size,
        n_hidden=args.hidden_dim,
        device=device,
    ).to(device)

    config = TrainConfig(
        hidden_dim=args.hidden_dim,
        lr=lr,
        epochs=epochs,
        l2_reg=args.l2_reg,
        lambda_orth=lambda_orth,
        lambda_tss=lambda_tss,
        log_freq=args.log_freq,
        checkpoint_dir=exp_checkpoint_dir,
        log_dir=args.log_dir,
        exp_name=exp_name,
        output_dir=exp_output_dir,
        infer_threshold_policy=args.infer_threshold_policy,  # type: ignore[arg-type]
    )

    trainer = THOCTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        eval_stride=test_stride,
        config=config,
        device=device,
        logger=logger,
    )

    if args.eval_only:
        checkpoint_path = args.checkpoint or os.path.join(config.checkpoint_dir, "best.pt")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        trainer.load(checkpoint_path)
        # train_history.json 에 저장된 validation 메타 복원 (infer 참고 지표용)
        history_path = os.path.join(exp_output_dir, "train_history.json")
        if os.path.isfile(history_path):
            with open(history_path, encoding="utf-8") as file:
                history = json.load(file)
            trainer.state.best_val_f1_pa = history.get("best_val_f1_pa", -1.0)
            trainer.state.best_threshold = history.get("best_threshold")
            logger.info(
                "Restored from history | best_val_f1_pa=%.4f | best_threshold=%s",
                trainer.state.best_val_f1_pa,
                trainer.state.best_threshold,
            )
        logger.info("Eval only | checkpoint: %s", checkpoint_path)
    else:
        trainer.train()
        checkpoint_path = os.path.join(config.checkpoint_dir, "best.pt")
        trainer.load(checkpoint_path)

    results = trainer.infer(window_size=window_size, test_stride=test_stride)
    results["best_val_f1_pa"] = trainer.state.best_val_f1_pa
    return results


def tune_hyperparameters(
    args: argparse.Namespace,
    logger: logging.Logger,
    device: torch.device,
) -> dict[str, float]:
    """그리드 서치로 lr, λ_orth, λ_TSS 탐색 후 최적 조합으로 full epoch 재학습."""
    combos = list(
        itertools.product(
            TUNE_GRID["lr"],
            TUNE_GRID["lambda_orth"],
            TUNE_GRID["lambda_tss"],
        )
    )
    logger.info("Hyperparameter tuning | %d combinations | tune_epochs=%d", len(combos), args.tune_epochs)

    best: dict | None = None
    tune_results: list[dict] = []

    for idx, (lr, lambda_orth, lambda_tss) in enumerate(combos, start=1):
        suffix = f"lr{lr:g}_orth{lambda_orth:g}_tss{lambda_tss:g}"
        exp_name = build_exp_name(
            args.dataset,
            args.window_size or get_dataset_defaults(args.dataset).window_size,
            args.tune_epochs,
            suffix,
        )
        logger.info(
            "=== Tune %d/%d | lr=%g | lambda_orth=%g | lambda_tss=%g ===",
            idx,
            len(combos),
            lr,
            lambda_orth,
            lambda_tss,
        )
        results = run_training(
            args,
            logger,
            device,
            lr=lr,
            lambda_orth=lambda_orth,
            lambda_tss=lambda_tss,
            epochs=args.tune_epochs,
            exp_name=exp_name,
        )
        entry = {
            "lr": lr,
            "lambda_orth": lambda_orth,
            "lambda_tss": lambda_tss,
            "best_val_f1_pa": results["best_val_f1_pa"],
            "test_f1_pa": results["f1_pa"],
            "test_f1": results["f1"],
            "exp_name": exp_name,
        }
        tune_results.append(entry)

        if best is None or entry["best_val_f1_pa"] > best["best_val_f1_pa"]:
            best = entry

    assert best is not None
    logger.info(
        "Best tune combo | lr=%g | lambda_orth=%g | lambda_tss=%g | val_f1_pa=%.4f | test_f1_pa=%.4f",
        best["lr"],
        best["lambda_orth"],
        best["lambda_tss"],
        best["best_val_f1_pa"],
        best["test_f1_pa"],
    )

    tune_summary_dir = os.path.join(args.output_dir, f"tune_{args.dataset.lower()}")
    os.makedirs(tune_summary_dir, exist_ok=True)
    save_json(os.path.join(tune_summary_dir, "tune_results.json"), tune_results)
    save_json(os.path.join(tune_summary_dir, "best_params.json"), best)

    logger.info(
        "=== Final training with best hyperparameters (%d epochs) ===",
        args.epochs,
    )
    final_exp = build_exp_name(
        args.dataset,
        args.window_size or get_dataset_defaults(args.dataset).window_size,
        args.epochs,
        f"best_lr{best['lr']:g}_orth{best['lambda_orth']:g}_tss{best['lambda_tss']:g}",
    )
    return run_training(
        args,
        logger,
        device,
        lr=best["lr"],
        lambda_orth=best["lambda_orth"],
        lambda_tss=best["lambda_tss"],
        epochs=args.epochs,
        exp_name=final_exp,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    defaults = get_dataset_defaults(args.dataset)
    window_size = args.window_size or defaults.window_size

    exp_name = args.exp_name or build_exp_name(args.dataset, window_size, args.epochs)
    logger = setup_logger(
        name="thoc",
        log_dir=args.log_dir,
        exp_name=exp_name,
        level=getattr(logging, args.log_level),
    )

    device = resolve_device(args.device)
    logger.info("Experiment: %s", exp_name)
    logger.info("Dataset: %s", args.dataset)
    logger.info(
        "Config | window=%d | val_ratio=%.2f | val_split=%s | epochs=%d | device=%s | eval_only=%s",
        window_size,
        args.val_ratio,
        not args.no_val_split,
        args.epochs,
        device,
        args.eval_only,
    )

    if args.tune:
        results = tune_hyperparameters(args, logger, device)
    elif args.eval_only and not args.exp_name and not args.checkpoint:
        raise SystemExit(
            "eval_only 모드에서는 --exp_name 또는 --checkpoint 를 지정해야 합니다."
        )
    else:
        results = run_training(
            args,
            logger,
            device,
            lr=args.lr,
            lambda_orth=args.lambda_orth,
            lambda_tss=args.lambda_tss,
            epochs=args.epochs,
            exp_name=exp_name,
        )

    logger.info("=== Final Evaluation ===")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info("%s: %.4f", key, value)
        else:
            logger.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
