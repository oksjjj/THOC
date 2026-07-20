"""
THOC 학습/추론 파이프라인.

전체 흐름:
  train()  → 정상 데이터로 L_total 최소화 (식 13)
             매 epoch validation F1-PA 로 best checkpoint 선택 (논문 §4.1)
  infer()  → validation threshold 로 test 전체 평가 (논문 §4.1 / §4.2)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from thoc.logger import log_banner, log_evaluation_summary, save_json
from thoc.metrics import ThresholdMethod, evaluate_anomaly_detection, select_threshold
from thoc.model import THOC

InferThresholdPolicy = Literal["test_best_f1_pa", "validation"]


@dataclass
class TrainConfig:
    hidden_dim: int = 128
    lr: float = 1e-3
    epochs: int = 30
    l2_reg: float = 1.0          # 식 (9) Ω(W) — weight decay
    lambda_orth: float = 1.0     # 식 (13) λ_orth
    lambda_tss: float = 1.0      # 식 (13) λ_TSS
    log_freq: int = 10
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    exp_name: str = "thoc"
    output_dir: str = "./outputs"
    threshold_method: ThresholdMethod = "best_f1_pa"
    n_threshold_steps: int = 500
    # 추론: validation 에서 고른 threshold 로 test 전체 평가 (논문 §4.2)
    infer_threshold_policy: InferThresholdPolicy = "validation"


@dataclass
class EpochStats:
    epoch: int
    avg_loss: float
    avg_thoc: float = 0.0
    avg_orth: float = 0.0
    avg_tss: float = 0.0
    val_f1_pa: float | None = None
    val_threshold: float | None = None


@dataclass
class TrainerState:
    best_loss: float = field(default=float("inf"))
    best_val_f1_pa: float = field(default=-1.0)
    best_threshold: float | None = None
    history: list[EpochStats] = field(default_factory=list)


class THOCTrainer:
    def __init__(
        self,
        model: THOC,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: TrainConfig,
        device: torch.device,
        logger: logging.Logger | None = None,
        val_loader: DataLoader | None = None,
        eval_stride: int | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.eval_stride = eval_stride
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger("thoc")
        self.state = TrainerState()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.l2_reg,
        )
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)

    def train(self) -> float:
        """
        전체 epoch 학습.

        validation 이 있으면 매 epoch validation F1-PA 기준으로 best.pt 저장.
        없으면 train loss 기준으로 저장 (레거시).
        """
        self.logger.info(
            "Training started: epochs=%d batches=%d val=%s",
            self.config.epochs,
            len(self.train_loader),
            "yes" if self.val_loader is not None else "no",
        )

        for epoch in range(1, self.config.epochs + 1):
            stats = self._train_epoch(epoch)

            if self.val_loader is not None:
                val_metrics = self._validate(window_size=self.model.window_size)
                stats.val_f1_pa = val_metrics["f1_pa"]
                stats.val_threshold = val_metrics["threshold"]
                self.logger.info(
                    "epoch=%d/%d loss=%.4f L_THOC=%.4f L_orth=%.4f L_TSS=%.4f "
                    "val_f1_pa=%.4f val_threshold=%.4f",
                    epoch,
                    self.config.epochs,
                    stats.avg_loss,
                    stats.avg_thoc,
                    stats.avg_orth,
                    stats.avg_tss,
                    val_metrics["f1_pa"],
                    val_metrics["threshold"],
                )
                if val_metrics["f1_pa"] > self.state.best_val_f1_pa:
                    self.state.best_val_f1_pa = val_metrics["f1_pa"]
                    self.state.best_threshold = val_metrics["threshold"]
                    checkpoint_path = os.path.join(self.config.checkpoint_dir, "best.pt")
                    self.save(checkpoint_path)
                    self.logger.info(
                        "Best model saved to %s (val_f1_pa=%.4f)",
                        checkpoint_path,
                        val_metrics["f1_pa"],
                    )
            else:
                self.logger.info(
                    "epoch=%d/%d loss=%.4f L_THOC=%.4f L_orth=%.4f L_TSS=%.4f",
                    epoch,
                    self.config.epochs,
                    stats.avg_loss,
                    stats.avg_thoc,
                    stats.avg_orth,
                    stats.avg_tss,
                )
                if stats.avg_loss < self.state.best_loss:
                    self.state.best_loss = stats.avg_loss
                    checkpoint_path = os.path.join(self.config.checkpoint_dir, "best.pt")
                    self.save(checkpoint_path)
                    self.logger.info("Best model saved to %s", checkpoint_path)

            self.state.history.append(stats)

        history_path = os.path.join(self.config.output_dir, "train_history.json")
        save_json(
            history_path,
            {
                "best_loss": self.state.best_loss,
                "best_val_f1_pa": self.state.best_val_f1_pa,
                "best_threshold": self.state.best_threshold,
                "lambda_orth": self.config.lambda_orth,
                "lambda_tss": self.config.lambda_tss,
                "lr": self.config.lr,
                "epochs": [
                    {
                        "epoch": s.epoch,
                        "avg_loss": s.avg_loss,
                        "avg_thoc": s.avg_thoc,
                        "avg_orth": s.avg_orth,
                        "avg_tss": s.avg_tss,
                        "val_f1_pa": s.val_f1_pa,
                        "val_threshold": s.val_threshold,
                    }
                    for s in self.state.history
                ],
            },
        )
        self.logger.info("Training history saved: %s", history_path)
        return self.state.best_val_f1_pa if self.val_loader else self.state.best_loss

    def _train_epoch(self, epoch: int) -> EpochStats:
        self.model.train()

        total_loss = 0.0
        total_thoc = 0.0
        total_orth = 0.0
        total_tss = 0.0
        log_interval = max(1, len(self.train_loader) // self.config.log_freq)

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for step, (batch_x, _) in enumerate(progress, start=1):
            batch_x = batch_x.to(self.device)
            _, losses = self.model(batch_x)

            loss = (
                losses["L_THOC"]
                + self.config.lambda_orth * losses["L_orth"]
                + self.config.lambda_tss * losses["L_TSS"]
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_thoc += losses["L_THOC"].item()
            total_orth += losses["L_orth"].item()
            total_tss += losses["L_TSS"].item()

            if step % log_interval == 0:
                progress.set_postfix(
                    total=f"{loss.item():.4f}",
                    thoc=f"{losses['L_THOC'].item():.4f}",
                )

        n_batches = len(self.train_loader)
        return EpochStats(
            epoch=epoch,
            avg_loss=total_loss / n_batches,
            avg_thoc=total_thoc / n_batches,
            avg_orth=total_orth / n_batches,
            avg_tss=total_tss / n_batches,
        )

    @torch.no_grad()
    def _validate(self, window_size: int) -> dict[str, float]:
        """Validation set 에서 best F1-PA threshold 와 해당 성능 산출."""
        if self.val_loader is None:
            raise RuntimeError("val_loader is not configured")

        self.model.eval()
        labels = self.val_loader.dataset.y
        scores = self._compute_point_scores(
            self.val_loader,
            window_size=window_size,
            stride=self.eval_stride,
        )
        threshold, _ = select_threshold(
            labels,
            scores,
            method=self.config.threshold_method,
            n_steps=self.config.n_threshold_steps,
        )
        return evaluate_anomaly_detection(
            labels,
            scores,
            threshold=threshold,
            threshold_method=self.config.threshold_method,
        )

    @torch.no_grad()
    def infer(
        self,
        window_size: int,
        test_stride: int | None = None,
        threshold: float | None = None,
    ) -> dict[str, float]:
        """
        Test set 평가 (논문: test 전체, threshold 는 validation 에서 선택).

        threshold 정책 (config.infer_threshold_policy):
          - validation (기본): 학습 중 validation F1-PA 최적 threshold → test 전체 적용
          - test_best_f1_pa: test 라벨로 threshold 재탐색 (참고·벤치마크 비교용)

        test_best_f1_pa (참고 지표) 는 results.json 에 저장.
        """
        self.model.eval()
        log_banner(self.logger, "testing", char="-")

        labels = self.test_loader.dataset.y
        scores = self._compute_point_scores(
            self.test_loader,
            window_size=window_size,
            stride=test_stride if test_stride is not None else self.eval_stride,
        )

        val_threshold = self.state.best_threshold
        extra: dict[str, float | dict[str, float]] = {
            "best_val_f1_pa": self.state.best_val_f1_pa,
        }

        # test 에서 threshold 재탐색 시 성능 (참고용, 논문 기본 프로토콜 아님)
        test_threshold, _ = select_threshold(
            labels,
            scores,
            method=self.config.threshold_method,
            n_steps=self.config.n_threshold_steps,
        )
        test_tuned = evaluate_anomaly_detection(
            labels,
            scores,
            threshold=test_threshold,
        )
        extra["test_best_f1_pa_threshold"] = test_threshold
        extra["test_best_f1_pa_metrics"] = {
            k: test_tuned[k]
            for k in (
                "precision",
                "recall",
                "f1",
                "f1_pa",
                "precision_pa",
                "recall_pa",
            )
        }

        if val_threshold is not None:
            extra["val_threshold"] = val_threshold

        if threshold is not None:
            threshold_source = "manual"
        elif self.config.infer_threshold_policy == "validation" and val_threshold is not None:
            threshold = val_threshold
            threshold_source = "validation"
        else:
            threshold, _ = select_threshold(
                labels,
                scores,
                method=self.config.threshold_method,
                n_steps=self.config.n_threshold_steps,
            )
            threshold_source = "test_best_f1_pa"

        results = evaluate_anomaly_detection(
            labels,
            scores,
            threshold=threshold,
            threshold_method=self.config.threshold_method,
            n_threshold_steps=self.config.n_threshold_steps,
        )
        results["threshold_source"] = threshold_source

        if val_threshold is not None and threshold_source == "validation":
            n_pred = int((scores > val_threshold).sum())
            if n_pred == 0 and labels.sum() > 0:
                self.logger.warning(
                    "Validation threshold %.4f flags 0/%d test points "
                    "(test score max=%.4f).",
                    val_threshold,
                    len(scores),
                    float(scores.max()),
                )

        primary_label = threshold_source
        if threshold_source == "validation":
            primary_note = "primary metric (threshold from validation)"
        elif threshold_source == "manual":
            primary_note = "manual threshold"
        else:
            primary_note = "oracle upper bound (threshold fit on test labels)"

        primary_metrics = {
            "F1-PA": results["f1_pa"],
            "Precision-PA": results["precision_pa"],
            "Recall-PA": results["recall_pa"],
            "F1": results["f1"],
            "Precision": results["precision"],
            "Recall": results["recall"],
            "AUC": results["auc"],
            "Threshold": results["threshold"],
            "Source": threshold_source,
        }

        reference_metrics = None
        reference_label = None
        reference_note = None
        if threshold_source == "validation":
            reference_label = "test_best_f1_pa"
            reference_note = "oracle upper bound (threshold fit on test labels)"
            reference_metrics = {
                "F1-PA": test_tuned["f1_pa"],
                "Precision-PA": test_tuned["precision_pa"],
                "Recall-PA": test_tuned["recall_pa"],
                "F1": test_tuned["f1"],
                "Precision": test_tuned["precision"],
                "Recall": test_tuned["recall"],
                "Threshold": test_threshold,
            }

        training_info: dict[str, float | str] = {
            "best_val_f1_pa": self.state.best_val_f1_pa,
            "best_threshold": (
                f"{self.state.best_threshold:.4f}"
                if self.state.best_threshold is not None
                else "N/A"
            ),
            "lr": self.config.lr,
            "lambda_orth": self.config.lambda_orth,
            "lambda_tss": self.config.lambda_tss,
            "checkpoint_dir": self.config.checkpoint_dir,
        }

        log_evaluation_summary(
            self.logger,
            primary=primary_metrics,
            primary_label=primary_label,
            primary_note=primary_note,
            reference=reference_metrics,
            reference_label=reference_label,
            reference_note=reference_note,
            training=training_info,
        )

        results_path = os.path.join(self.config.output_dir, "results.json")
        save_json(results_path, {**results, **extra})
        self.logger.info("Results saved: %s", results_path)
        return results

    @torch.no_grad()
    def _compute_point_scores(
        self,
        loader: DataLoader,
        window_size: int,
        stride: int | None = None,
    ) -> np.ndarray:
        stride = stride if stride is not None else loader.dataset.stride

        window_scores: list[torch.Tensor] = []
        for batch_x, _ in tqdm(loader, desc="Scoring", leave=False):
            batch_x = batch_x.to(self.device)
            anomaly_scores, _ = self.model(batch_x)
            window_scores.append(anomaly_scores.cpu())

        scores = torch.cat(window_scores).numpy()

        if stride == 1:
            prefix = np.repeat(scores[0], window_size - 1)
            return np.concatenate([prefix, scores])

        expanded = np.zeros(len(loader.dataset.y), dtype=np.float32)
        for idx, score in enumerate(scores):
            start = idx * stride
            end = min(start + stride, len(expanded))
            expanded[start:end] = score

        if len(expanded) > len(scores) * stride:
            expanded[len(scores) * stride :] = scores[-1]

        return expanded

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.logger.info("Loading checkpoint: %s", path)
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
