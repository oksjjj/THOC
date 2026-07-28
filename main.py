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
    is_smd_machine,
    list_smd_machines,
    load_dataset,
    make_dataset_splits,
    resolve_val_ratio,
)
from thoc.logger import log_banner, log_configurations, save_json, setup_logger
from thoc.model import THOC
from thoc.trainer import THOCTrainer, TrainConfig


KNOWN_DATASETS = ("NeurIPS-TS-UNI", "NeurIPS-TS-MUL", "SMAP", "SMD")


def parse_dataset_name(value: str) -> str:
    if value in KNOWN_DATASETS or is_smd_machine(value):
        return value
    raise argparse.ArgumentTypeError(
        f"Unknown dataset '{value}'. "
        f"Choose from {list(KNOWN_DATASETS)} or machine-{{g}}-{{id}} (e.g. machine-1-1)"
    )


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
        type=parse_dataset_name,
        default="NeurIPS-TS-MUL",
        help="NeurIPS-TS-UNI|MUL, SMAP, SMD(전체 concat), 또는 machine-1-1 등 SMD entity",
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--test_stride", type=int, default=None)
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=None,
        help="validation 비율 (SMAP/SMD: test 앞, NeurIPS-TS: abnormal 앞; 기본 SMAP/SMD=0.15, NeurIPS=0.3)",
    )
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
    parser.add_argument(
        "--save_dir",
        "--checkpoint_dir",
        type=str,
        default="./model",
        dest="save_dir",
        help="체크포인트 루트 (OmniAnomaly save_dir; 기본 ./model)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./log",
        help="로그 루트 (OmniAnomaly log_dir; 기본 ./log)",
    )
    parser.add_argument(
        "--result_dir",
        "--output_dir",
        type=str,
        default="./result",
        dest="result_dir",
        help="결과/메트릭 루트 (OmniAnomaly result_dir; 기본 ./result)",
    )
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
        help="eval_only 시 사용할 .pt 경로 (미지정 시 model/{exp_name}/best.pt)",
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
    parser.add_argument(
        "--all_smd_machines",
        action="store_true",
        help="data/SMD 의 모든 machine-* 에 대해 순차 학습·평가 "
        "(OmniAnomaly 머신별 실험; --dataset 은 무시되고 machine id 사용)",
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

    exp_save_dir = os.path.join(args.save_dir, exp_name)
    exp_result_dir = os.path.join(args.result_dir, exp_name)

    train_x, train_y, test_x, test_y = load_dataset(args.dataset, data_dir)
    val_ratio = resolve_val_ratio(args.dataset, args.val_ratio)

    if args.no_val_split:
        splits = make_dataset_splits(
            args.dataset, train_x, train_y, test_x, test_y, val_ratio=0.0
        )
        val_x, val_y = None, None
    else:
        splits = make_dataset_splits(
            args.dataset, train_x, train_y, test_x, test_y, val_ratio=val_ratio
        )
        val_x, val_y = splits.val_x, splits.val_y

    logger.info(
        "train set shape: %s | val: %s | test: %s",
        splits.train_x.shape,
        None if val_x is None else val_x.shape,
        splits.test_x.shape,
    )
    if val_y is not None:
        logger.info(
            "anomaly ratio: val=%.2f%% test=%.2f%%",
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
        "train windows: %d | val windows: %s | test windows: %d",
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
        checkpoint_dir=exp_save_dir,
        log_dir=args.log_dir,
        exp_name=exp_name,
        output_dir=exp_result_dir,
        dataset=args.dataset,
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
        history_path = os.path.join(exp_result_dir, "train_history.json")
        if os.path.isfile(history_path):
            with open(history_path, encoding="utf-8") as file:
                history = json.load(file)
            trainer.state.best_val_f1_pa = history.get("best_val_f1_pa", -1.0)
            trainer.state.best_threshold = history.get("best_threshold")
            logger.info(
                "restored from history: best_val_f1_pa=%.4f best_threshold=%s",
                trainer.state.best_val_f1_pa,
                trainer.state.best_threshold,
            )
        logger.info("eval only: checkpoint=%s", checkpoint_path)
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
    logger.info(
        "hyperparameter tuning: combinations=%d tune_epochs=%d",
        len(combos),
        args.tune_epochs,
    )

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
        log_banner(logger, f"Tune {idx}/{len(combos)}")
        logger.info(
            "lr=%g lambda_orth=%g lambda_tss=%g",
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
        "best tune combo: lr=%g lambda_orth=%g lambda_tss=%g "
        "val_f1_pa=%.4f test_f1_pa=%.4f",
        best["lr"],
        best["lambda_orth"],
        best["lambda_tss"],
        best["best_val_f1_pa"],
        best["test_f1_pa"],
    )

    tune_summary_dir = os.path.join(args.result_dir, f"tune_{args.dataset.lower()}")
    os.makedirs(tune_summary_dir, exist_ok=True)
    save_json(os.path.join(tune_summary_dir, "tune_results.json"), tune_results)
    save_json(os.path.join(tune_summary_dir, "best_params.json"), best)

    log_banner(logger, "Final training")
    logger.info("epochs=%d (best hyperparameters)", args.epochs)
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


def run_single_experiment(args: argparse.Namespace, device: torch.device | None = None) -> dict[str, float] | None:
    """단일 dataset/machine 에 대한 학습·평가 (또는 tune)."""
    set_seed(args.seed)

    defaults = get_dataset_defaults(args.dataset)
    window_size = args.window_size or defaults.window_size

    exp_name = args.exp_name or build_exp_name(args.dataset, window_size, args.epochs)
    logger = setup_logger(
        name=f"thoc.{args.dataset}",
        log_dir=args.log_dir,
        exp_name=exp_name,
        level=getattr(logging, args.log_level),
    )

    device = device or resolve_device(args.device)
    logger.info("Experiment started")
    logger.info("Experiment: %s", exp_name)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Using device: %s", device)

    config_dump = {
        "batch_size": args.batch_size,
        "checkpoint": args.checkpoint,
        "save_dir": args.save_dir,
        "data_dir": args.data_dir or defaults.data_dir,
        "dataset": args.dataset,
        "device": str(device),
        "epochs": args.epochs,
        "eval_batch_size": args.eval_batch_size,
        "eval_only": args.eval_only,
        "exp_name": exp_name,
        "hidden_dim": args.hidden_dim,
        "infer_threshold_policy": args.infer_threshold_policy,
        "l2_reg": args.l2_reg,
        "lambda_orth": args.lambda_orth,
        "lambda_tss": args.lambda_tss,
        "log_dir": args.log_dir,
        "log_freq": args.log_freq,
        "lr": args.lr,
        "result_dir": args.result_dir,
        "scaler": args.scaler,
        "seed": args.seed,
        "stride": args.stride,
        "test_stride": args.test_stride,
        "tune": args.tune,
        "tune_epochs": args.tune_epochs,
        "val_ratio": resolve_val_ratio(args.dataset, args.val_ratio),
        "val_split": not args.no_val_split,
        "window_size": window_size,
    }
    log_configurations(logger, dict(sorted(config_dump.items())))

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

    logger.info("Experiment finished")
    return results


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.all_smd_machines:
        data_dir = args.data_dir or get_dataset_defaults("SMD").data_dir
        machines = list_smd_machines(data_dir)
        if not machines:
            raise SystemExit(
                f"No machine-* files under {data_dir}. "
                "Run: python scripts/preprocess_data.py --dataset SMD "
                "--raw_dir <ServerMachineDataset> --output_dir data/SMD"
            )
        if args.tune:
            raise SystemExit("--all_smd_machines 와 --tune 은 함께 쓸 수 없습니다.")

        summary_name = f"smd_all_machines_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        summary_logger = setup_logger(
            name="thoc.smd_all",
            log_dir=args.log_dir,
            exp_name=summary_name,
            level=getattr(logging, args.log_level),
        )
        summary_logger.info(
            "Training all SMD machines | count=%d | data_dir=%s",
            len(machines),
            data_dir,
        )
        summary_logger.info("Machines: %s", ", ".join(machines))

        summary_rows: list[dict] = []
        for idx, machine in enumerate(machines, start=1):
            summary_logger.info("=== [%d/%d] %s ===", idx, len(machines), machine)
            machine_args = argparse.Namespace(**vars(args))
            machine_args.dataset = machine
            machine_args.exp_name = None  # machine 별 새 exp
            machine_args.all_smd_machines = False
            try:
                results = run_single_experiment(machine_args, device=device)
                row = {
                    "machine": machine,
                    "status": "ok",
                    "f1_pa": None if results is None else results.get("f1_pa"),
                    "auroc": None if results is None else results.get("auroc"),
                    "auprc": None if results is None else results.get("auprc"),
                    "threshold": None if results is None else results.get("threshold"),
                }
            except Exception as exc:  # noqa: BLE001 — 머신별 실패 시 다음으로
                summary_logger.exception("Failed on %s: %s", machine, exc)
                row = {
                    "machine": machine,
                    "status": f"error: {exc}",
                    "f1_pa": None,
                    "auroc": None,
                    "auprc": None,
                    "threshold": None,
                }
            summary_rows.append(row)

        summary_path = os.path.join(args.result_dir, summary_name, "smd_machine_summary.json")
        save_json(summary_path, summary_rows)
        summary_logger.info("SMD machine summary saved: %s", summary_path)
        for row in summary_rows:
            summary_logger.info(
                "%s | status=%s | f1_pa=%s | auroc=%s | auprc=%s",
                row["machine"],
                row["status"],
                row["f1_pa"],
                row["auroc"],
                row["auprc"],
            )
        summary_logger.info("Experiment finished")
        return

    run_single_experiment(args, device=device)


if __name__ == "__main__":
    main()
