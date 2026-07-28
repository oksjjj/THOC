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
import logging
import os
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

_logger = logging.getLogger("thoc")

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


def confusion_counts(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, int]:
    """TP / TN / FP / FN from binary predictions."""
    labels = labels.astype(int)
    predictions = predictions.astype(int)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def detection_latency(labels: np.ndarray, predictions: np.ndarray) -> float:
    """
    이상 구간 시작부터 첫 탐지까지 평균 지연(스텝).

    각 GT 이상 구간에서 prediction==1 이 처음 나타난 시점까지의 거리를 평균.
    """
    delays: list[int] = []
    idx = 0
    n = len(labels)
    while idx < n:
        if labels[idx] != 1:
            idx += 1
            continue

        seg_start = idx
        detected = False
        while idx < n and labels[idx] == 1:
            if predictions[idx] == 1:
                delays.append(idx - seg_start)
                detected = True
                break
            idx += 1
        while idx < n and labels[idx] == 1:
            idx += 1
        if not detected:
            continue

    return float(np.mean(delays)) if delays else 0.0


def _prepare_rank_inputs(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels).reshape(-1).astype(bool)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim > 1:
        scores = scores.sum(axis=-1)
    scores = scores.reshape(-1)
    if len(y_true) != len(scores):
        raise ValueError("scores and labels must have the same length")
    return scores, y_true


def _anomaly_segment_bounds(y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (starts, ends) half-open index ranges for contiguous anomaly segments."""
    actual = np.asarray(y_true, dtype=bool).reshape(-1)
    padded = np.concatenate([[False], actual, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return starts, ends


def _point_adjusted_scores(
    scores: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """
    Convert raw scores to point-adjustment-equivalent scores.

    PA 에서는 이상 구간 내 한 점이라도 threshold 를 넘으면 구간 전체가 탐지된다.
    THOC 는 점수가 클수록 이상이므로, 각 구간을 그 구간의 최대 점수(가장 이상한
    점수)로 치환하면 sklearn ranking 함수가 모든 threshold 에서 PA 혼동행렬을
    정확히 재현한다. 정상 구간 점수는 그대로 둔다.
    """
    adj = np.asarray(scores, dtype=float).copy()
    for a, b in zip(starts, ends):
        adj[a:b] = adj[a:b].max()
    return adj


def calc_pa_curves(
    scores: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, Any]:
    """
    ROC / PR curves and AUC scores with point adjustment, via scikit-learn
    (OmniAnomaly `_calc_pa_curves`).

    ``scores`` / ``y_true`` must already be prepared by ``_prepare_rank_inputs``.
    THOC 는 점수가 클수록 이상이므로 조정된 점수를 그대로 sklearn 에 입력한다.
    """
    starts, ends = _anomaly_segment_bounds(y_true)
    y_score = _point_adjusted_scores(scores, starts, ends)

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "fpr": fpr,
        "tpr": tpr,
        "recall": recall,
        "precision": precision,
        "prevalence": float(y_true.mean()),
    }


def save_roc_pr_curves(
    curve: dict[str, Any],
    save_dir: str,
    prefix: str = "roc_pr",
    dataset: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, str]:
    """
    PA 기준 ROC / PR 곡선 이미지 저장 (OmniAnomaly `save_roc_pr_curves`).

    Writes:
      ``{save_dir}/{prefix}_roc.png``
      ``{save_dir}/{prefix}_pr.png``
      ``{save_dir}/{prefix}_combined.png``
    """
    os.makedirs(save_dir, exist_ok=True)
    title_ds = f" ({dataset})" if dataset else ""
    auroc = curve["auroc"]
    auprc = curve["auprc"]
    prevalence = curve.get("prevalence", 0.0)
    log = logger or _logger
    paths: dict[str, str] = {}

    # ROC (sklearn RocCurveDisplay)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay(
        fpr=curve["fpr"], tpr=curve["tpr"], roc_auc=auroc,
    ).plot(ax=ax, name="THOC (PA)", plot_chance_level=True,
           curve_kwargs={"color": "#1f77b4", "lw": 2})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"ROC curve — point adjustment{title_ds}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    roc_path = os.path.join(save_dir, f"{prefix}_roc.png")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    paths["roc_curve"] = roc_path

    # PR (sklearn PrecisionRecallDisplay)
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay(
        precision=curve["precision"], recall=curve["recall"],
        average_precision=auprc,
    ).plot(ax=ax, name="THOC (PA)",
           curve_kwargs={"color": "#d62728", "lw": 2})
    ax.axhline(prevalence, color="k", ls="--", lw=1,
               label=f"prevalence={prevalence:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"PR curve — point adjustment{title_ds}")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    pr_path = os.path.join(save_dir, f"{prefix}_pr.png")
    fig.tight_layout()
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    paths["pr_curve"] = pr_path

    # Combined
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay(
        fpr=curve["fpr"], tpr=curve["tpr"], roc_auc=auroc,
    ).plot(ax=axes[0], name="PA", plot_chance_level=True,
           curve_kwargs={"color": "#1f77b4", "lw": 2})
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title(f"ROC — PA{title_ds}")
    axes[0].grid(True, alpha=0.3)

    PrecisionRecallDisplay(
        precision=curve["precision"], recall=curve["recall"],
        average_precision=auprc,
    ).plot(ax=axes[1], name="PA",
           curve_kwargs={"color": "#d62728", "lw": 2})
    axes[1].axhline(prevalence, color="k", ls="--", lw=1,
                    label=f"prevalence={prevalence:.4f}")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title(f"PR — PA{title_ds}")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    combined_path = os.path.join(save_dir, f"{prefix}_combined.png")
    fig.tight_layout()
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    paths["roc_pr_combined"] = combined_path

    log.info("ROC curve saved to %s", roc_path)
    log.info("PR curve saved to %s", pr_path)
    log.info("Combined curves saved to %s", combined_path)
    return paths


def calc_rank_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    save_dir: str | None = None,
    dataset: str | None = None,
    prefix: str = "roc_pr",
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    AUROC / AUPRC with point adjustment, computed via scikit-learn
    (OmniAnomaly `calc_rank_metrics`).

    각 이상 구간을 구간 내 최대 점수로 치환한 PA-equivalent score 를 sklearn
    ranking 함수에 입력해, 모든 threshold 에서 PA 혼동행렬을 정확히 재현한다.
    AUPRC 는 sklearn Average Precision. ``save_dir`` 이 있으면 ROC/PR 이미지를
    저장한다.
    """
    scores, y_true = _prepare_rank_inputs(scores, labels)
    if len(np.unique(y_true)) < 2:
        nan = float("nan")
        return {
            "auroc": nan,
            "auprc": nan,
            "auroc_pa": nan,
            "auprc_pa": nan,
            "point_adjustment": True,
        }

    curve = calc_pa_curves(scores, y_true)
    out: dict[str, Any] = {
        "auroc": float(curve["auroc"]),
        "auprc": float(curve["auprc"]),
        "auroc_pa": float(curve["auroc"]),
        "auprc_pa": float(curve["auprc"]),
        "point_adjustment": True,
    }
    if save_dir is not None:
        paths = save_roc_pr_curves(
            curve,
            save_dir,
            prefix=prefix,
            dataset=dataset,
            logger=logger,
        )
        out.update(paths)
    return out


def pa_ranking_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """하위 호환 래퍼 — PA AUROC/AUPRC 숫자만 반환."""
    ranking = calc_rank_metrics(scores, labels)
    return {"auroc_pa": float(ranking["auroc_pa"]), "auprc_pa": float(ranking["auprc_pa"])}


def format_pa_classification_section(metrics: dict[str, Any]) -> dict[str, float | str]:
    """EVALUATION SUMMARY 분류 지표 블록 (모두 PA 기준)."""
    return {
        "F1": metrics["f1_pa"],
        "Precision": metrics["precision_pa"],
        "Recall": metrics["recall_pa"],
        "TP / FP": f"{int(metrics['tp'])} / {int(metrics['fp'])}",
        "TN / FN": f"{int(metrics['tn'])} / {int(metrics['fn'])}",
        "Threshold": metrics["threshold"],
        "Latency": metrics["latency"],
    }


def format_pa_ranking_section(metrics: dict[str, Any]) -> dict[str, float | str]:
    """EVALUATION SUMMARY Ranking 블록 (OmniAnomaly 동일 필드)."""
    section: dict[str, float | str] = {
        "AUROC": metrics.get("auroc_pa", metrics.get("auroc", float("nan"))),
        "AUPRC": metrics.get("auprc_pa", metrics.get("auprc", float("nan"))),
    }
    if metrics.get("roc_curve"):
        section["ROC image"] = str(metrics["roc_curve"])
    if metrics.get("pr_curve"):
        section["PR image"] = str(metrics["pr_curve"])
    if metrics.get("roc_pr_combined"):
        section["Combined"] = str(metrics["roc_pr_combined"])
    return section

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
    compute_ranking: bool = True,
    save_dir: str | None = None,
    dataset: str | None = None,
    curve_prefix: str = "roc_pr",
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    시점별 AnomalyScore 와 라벨로 최종 평가.

    Args:
        labels:    (N,) — 0=정상, 1=이상
        scores:    (N,) — AnomalyScore(x_t), 클수록 이상 가능성 높음
        threshold: δ. None 이면 threshold_method 로 자동 결정
        threshold_method: "best_f1_pa" (논문) 또는 "youden"
        compute_ranking: PA AUROC/AUPRC (및 선택적 곡선 저장)
        save_dir: ROC/PR 이미지 저장 디렉터리 (None 이면 숫자만)

    Returns:
        threshold, PA 분류 지표, auroc_pa/auprc_pa, (optional) curve paths
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
    pa_counts = confusion_counts(labels, pa_predictions)

    labels_arr = labels.astype(int)
    if len(np.unique(labels_arr)) < 2:
        auroc_raw = 0.5
        auprc_raw = 0.0
    else:
        auroc_raw = float(roc_auc_score(labels_arr, scores))
        auprc_raw = float(average_precision_score(labels_arr, scores))

    result: dict[str, Any] = {
        "threshold": float(threshold),
        "auc": auroc_raw,
        "auroc_raw": auroc_raw,
        "auprc_raw": auprc_raw,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy_pa": pa_metrics["accuracy"],
        "precision_pa": pa_metrics["precision"],
        "recall_pa": pa_metrics["recall"],
        "f1_pa": pa_metrics["f1"],
        "tp": float(pa_counts["tp"]),
        "tn": float(pa_counts["tn"]),
        "fp": float(pa_counts["fp"]),
        "fn": float(pa_counts["fn"]),
        "latency": detection_latency(labels, pa_predictions),
        "point_adjustment": True,
    }
    if val_f1_pa is not None:
        result["val_f1_pa_at_threshold"] = val_f1_pa

    if compute_ranking:
        ranking = calc_rank_metrics(
            scores,
            labels,
            save_dir=save_dir,
            dataset=dataset,
            prefix=curve_prefix,
            logger=logger,
        )
        # OmniAnomaly 와 동일: summary / results 의 auroc·auprc 는 PA 기준
        result["auroc"] = ranking["auroc"]
        result["auprc"] = ranking["auprc"]
        result["auroc_pa"] = ranking["auroc_pa"]
        result["auprc_pa"] = ranking["auprc_pa"]
        for key in ("roc_curve", "pr_curve", "roc_pr_combined"):
            if key in ranking:
                result[key] = ranking[key]

    return result
