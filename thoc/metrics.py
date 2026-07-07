"""
이상 탐지 평가 지표.

infer() 에서 _compute_point_scores() 로 얻은 점수 배열을
ground-truth 라벨과 비교해 최종 성능을 산출한다.

평가 흐름 (논문 §4.1 / §4.2):
  scores (시점별 AnomalyScore)
    → validation set 에서 best F1-PA threshold δ 탐색
    → test set 전체에 δ 적용
    → F1, Precision, Recall, F1-PA 계산
"""

from __future__ import annotations

import copy
from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ThresholdMethod = Literal["best_f1_pa", "youden"]


def point_adjust(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """
    Point-Adjust (PA) 보정.

    연속 이상 구간에서 1개 시점이라도 탐지하면 구간 전체를 정탐 처리.
    SMAP/MSL 벤치마크에서 표준으로 사용.
    """
    adjusted = copy.deepcopy(predictions)
    in_anomaly = False  # 현재 이상 구간 안에 있는지 여부

    for idx in range(len(labels)):
        if labels[idx] == 1 and adjusted[idx] == 1 and not in_anomaly:
            # 이상 구간에서 처음으로 탐지 성공
            in_anomaly = True
            # 구간 시작까지 소급: 앞쪽 미탐지 시점도 정탐으로 보정
            for j in range(idx, -1, -1):
                if labels[j] == 0:
                    break
                adjusted[j] = 1
            # 구간 끝까지 전방: 뒤쪽 미탐지 시점도 정탐으로 보정
            for j in range(idx, len(labels)):
                if labels[j] == 0:
                    break
                adjusted[j] = 1
        elif labels[idx] == 0:
            in_anomaly = False  # 정상 구간 진입 → 플래그 리셋
        elif in_anomaly:
            adjusted[idx] = 1  # 이상 구간 내부는 모두 정탐

    return adjusted


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Accuracy, Precision, Recall, F1 계산."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def best_f1_pa_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    n_steps: int = 500,
) -> tuple[float, float]:
    """
    Validation set 에서 F1-PA 를 최대화하는 threshold δ 탐색.

    OmniAnomaly bf_search 와 동일한 grid search 방식.
  논문: hyperparameter / threshold 는 validation F1 기준.
    """
    if len(scores) == 0:
        return 0.0, 0.0

    lo, hi = float(np.min(scores)), float(np.max(scores))
    if lo == hi:
        return lo, 0.0

    # grid 상한을 max 미만으로 두어 scores > threshold 가 성립 가능하게 함
    margin = max((hi - lo) * 0.01, 1e-6)
    thresholds = np.linspace(lo - margin, hi - margin, n_steps)
    best_f1 = -1.0
    best_threshold = hi

    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        pa_predictions = point_adjust(labels, predictions)
        f1_pa = f1_score(labels, pa_predictions, zero_division=0)
        if f1_pa > best_f1:
            best_f1 = float(f1_pa)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Youden's J statistic 으로 threshold δ 선택.

    J = TPR - FPR 을 최대화하는 δ.
    (레거시 / 비교용 — 논문 프로토콜은 best_f1_pa 권장)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr          # J = TPR - FPR
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    method: ThresholdMethod = "best_f1_pa",
    n_steps: int = 500,
) -> tuple[float, float | None]:
    """
    threshold 선택.

    Returns:
        (threshold, val_f1_pa) — method=youden 이면 val_f1_pa 는 None
    """
    if method == "best_f1_pa":
        threshold, val_f1_pa = best_f1_pa_threshold(labels, scores, n_steps=n_steps)
        return threshold, val_f1_pa
    return youden_threshold(labels, scores), None


def evaluate_anomaly_detection(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
    threshold_method: ThresholdMethod = "best_f1_pa",
    n_threshold_steps: int = 500,
) -> dict[str, float]:
    """
    시점별 AnomalyScore 와 라벨로 최종 평가.

    Args:
        labels:    (N,) — 0=정상, 1=이상
        scores:    (N,) — AnomalyScore(x_t), 클수록 이상 가능성 높음
        threshold: δ. None 이면 threshold_method 로 자동 결정
        threshold_method: "best_f1_pa" (논문) 또는 "youden"

    Returns:
        threshold, auc, f1, f1_pa 등
    """
    val_f1_pa: float | None = None
    if threshold is None:
        threshold, val_f1_pa = select_threshold(
            labels,
            scores,
            method=threshold_method,
            n_steps=n_threshold_steps,
        )

    # AnomalyScore(x_t) > δ → anomaly(1), else → normal(0)
    predictions = (scores > threshold).astype(int)

    metrics = classification_metrics(labels, predictions)

    pa_predictions = point_adjust(labels, predictions)
    pa_metrics = classification_metrics(labels, pa_predictions)

    result: dict[str, float] = {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(labels, scores)),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy_pa": pa_metrics["accuracy"],
        "precision_pa": pa_metrics["precision"],
        "recall_pa": pa_metrics["recall"],
        "f1_pa": pa_metrics["f1"],
    }
    if val_f1_pa is not None:
        result["val_f1_pa_at_threshold"] = val_f1_pa
    return result
