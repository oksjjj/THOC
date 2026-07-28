#!/usr/bin/env python3
"""
GT anomaly segment viewer (OmniAnomaly viz_gt_anomalies.py 대응).

각 ground-truth 이상 구간을 PNG 로 저장한다.
구간 양쪽에 정상 컨텍스트(기본 10× 구간 길이)를 포함한다.

Modes:
  --gt_only   : GT 밴드만 (학습 전 / 점수 불필요)
  default     : results.json threshold + test_score 로 TP/FP/FN 오버레이
                (point adjustment 없음 — OmniAnomaly 와 동일)

  TP : GT anomaly & predicted      → red filled circles
  FP : normal & predicted          → red triangles
  FN : GT anomaly & not predicted  → blue open circles

Examples:
  python scripts/viz_gt_anomalies.py --dataset SMD --gt_only
  python scripts/viz_gt_anomalies.py --dataset machine-1-1 --gt_only
  python scripts/viz_gt_anomalies.py --dataset machine-1-1 --exp_name machine-1-1_ws100_...
  python scripts/viz_gt_anomalies.py --dataset SMAP --gt_only
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
import numpy as np

# repo root 를 path 에 추가
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from thoc.data import (  # noqa: E402
    get_dataset_defaults,
    is_smd_machine,
    load_dataset,
    make_dataset_splits,
    resolve_val_ratio,
)

KNOWN_DATASETS = ("NeurIPS-TS-UNI", "NeurIPS-TS-MUL", "SMAP", "SMD")


def parse_dataset_name(value: str) -> str:
    if value in KNOWN_DATASETS or is_smd_machine(value):
        return value
    raise argparse.ArgumentTypeError(
        f"Unknown dataset '{value}'. "
        f"Choose from {list(KNOWN_DATASETS)} or machine-{{g}}-{{id}}"
    )


def _configure_pyplot():
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.concatenate([[False], mask, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts.tolist(), ends.tolist()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save GT anomaly plots (optionally with TP/FP/FN markers)",
    )
    parser.add_argument(
        "--dataset",
        type=parse_dataset_name,
        required=True,
        help="SMAP/SMD/NeurIPS 또는 machine-1-1 등 SMD entity",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="데이터 디렉터리 (기본: dataset defaults)",
    )
    parser.add_argument(
        "--gt_only",
        action="store_true",
        help="GT segments only (no scores/metrics required)",
    )
    parser.add_argument(
        "--context_mult",
        type=float,
        default=10.0,
        help="Normal context length multiplier on each side (default: 10)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output root (default: viz_gt if --gt_only else viz_pred)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Experiment output root that contains {exp_name}/",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name under --output_dir (required unless --gt_only "
        "or --run_dir is set)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Direct path to an experiment output dir "
        "(contains results.json, test_score.npy)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override threshold (default: results.json)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=None,
        help="validation 비율 (--gt_only 시 full test 사용, pred 모드에서 "
        "split 맞출 때 사용; 기본 dataset default)",
    )
    parser.add_argument(
        "--no_val_split",
        action="store_true",
        help="val 분할 없이 full test 사용 (학습 --no_val_split 과 맞춤)",
    )
    parser.add_argument(
        "--max_dims",
        type=int,
        default=None,
        help="플롯에 그릴 최대 채널 수 (SMD 38채널 축소용; 기본=전체)",
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=None,
        help="저장할 최대 GT 세그먼트 수 (기본=전체)",
    )
    return parser.parse_args()


def _resolve_run_dir(args: argparse.Namespace) -> str:
    if args.run_dir is not None:
        return args.run_dir
    if args.exp_name is None:
        raise SystemExit("--exp_name 또는 --run_dir 이 필요합니다 (pred 모드).")
    return os.path.join(args.output_dir, args.exp_name)


def _load_pred(
    run_dir: str,
    threshold_override: float | None,
) -> tuple[np.ndarray, np.ndarray, float, str, str]:
    score_path = os.path.join(run_dir, "test_score.npy")
    if not os.path.isfile(score_path):
        raise SystemExit(f"Missing test scores: {score_path}")
    score = np.load(score_path).astype(float).reshape(-1)

    if threshold_override is not None:
        thr = float(threshold_override)
        thr_src = "cli"
    else:
        metrics_path = os.path.join(run_dir, "results.json")
        if not os.path.isfile(metrics_path):
            raise SystemExit(f"Missing results.json: {metrics_path}")
        with open(metrics_path, encoding="utf-8") as file:
            metrics = json.load(file)
        thr = float(metrics["threshold"])
        thr_src = metrics_path

    # THOC: higher score = more anomalous
    pred = score > thr
    return score, pred, thr, thr_src, score_path


def _load_series_for_gt(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Full test series/labels for --gt_only."""
    defaults = get_dataset_defaults(args.dataset)
    data_dir = args.data_dir or defaults.data_dir
    _, _, test_x, test_y = load_dataset(args.dataset, data_dir)
    return np.asarray(test_x, dtype=float), np.asarray(test_y).reshape(-1).astype(bool)


def _load_series_for_pred(
    args: argparse.Namespace,
    run_dir: str,
    n_score: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pred 모드용 시계열.

    우선순위:
      1) run_dir/test_x.npy + test_label.npy (infer 시 저장된 스케일된 시계열)
      2) 원본 데이터에서 val split 후 test 부분 (길이 맞춤)
    """
    series_path = os.path.join(run_dir, "test_x.npy")
    label_path = os.path.join(run_dir, "test_label.npy")
    if os.path.isfile(series_path) and os.path.isfile(label_path):
        x = np.load(series_path).astype(float)
        y = np.load(label_path).reshape(-1).astype(bool)
        if len(x) != len(y):
            raise SystemExit(
                f"Length mismatch in saved arrays: x={len(x)} y={len(y)}"
            )
        return x, y

    defaults = get_dataset_defaults(args.dataset)
    data_dir = args.data_dir or defaults.data_dir
    train_x, train_y, test_x, test_y = load_dataset(args.dataset, data_dir)
    val_ratio = resolve_val_ratio(args.dataset, args.val_ratio)
    if args.no_val_split:
        x = np.asarray(test_x, dtype=float)
        y = np.asarray(test_y).reshape(-1).astype(bool)
    else:
        splits = make_dataset_splits(
            args.dataset, train_x, train_y, test_x, test_y, val_ratio=val_ratio
        )
        x = np.asarray(splits.test_x, dtype=float)
        y = np.asarray(splits.test_y).reshape(-1).astype(bool)

    # sliding-window score 길이에 맞춤 (뒤쪽 정렬 — OmniAnomaly last-point 정렬과 유사)
    if len(x) > n_score:
        x = x[-n_score:]
        y = y[-n_score:]
    return x, y


def plot_gt_segment(
    x: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray | None,
    start: int,
    end: int,
    index: int,
    n_total: int,
    context_mult: float,
    threshold: float | None,
    out_path: str,
    gt_only: bool = False,
    max_dims: int | None = None,
) -> None:
    plt = _configure_pyplot()

    seg_len = max(end - start, 1)
    pad = int(round(seg_len * context_mult))
    left = max(0, start - pad)
    right = min(len(y), end + pad)
    xs = np.arange(left, right)
    x_win = x[left:right]
    y_win = y[left:right]
    n_dims = x.shape[1] if x.ndim > 1 else 1
    if max_dims is not None:
        n_dims = min(n_dims, max_dims)
    if x.ndim == 1:
        x_win = x_win.reshape(-1, 1)

    if gt_only or pred is None:
        pred_win = None
        n_tp = n_fp = n_fn = 0
        tp_mask = fp_mask = fn_mask = None
    else:
        pred_win = pred[left:right]
        tp_mask = y_win & pred_win
        fp_mask = (~y_win) & pred_win
        fn_mask = y_win & (~pred_win)
        n_tp = int(tp_mask.sum())
        n_fp = int(fp_mask.sum())
        n_fn = int(fn_mask.sum())

    row_h = 0.5
    fig_h = max(5.0, 0.8 + row_h * n_dims)
    fig, axes = plt.subplots(
        n_dims,
        1,
        figsize=(12, fig_h),
        sharex=True,
        gridspec_kw={"hspace": 0.06},
    )
    if n_dims == 1:
        axes = [axes]

    pink = "#f7b6c2"
    green = "#c8e6c9"
    line_c = "#1f4e79"
    red = "#d32f2f"
    fn_c = "#1565c0"

    if gt_only:
        title = (
            f"{n_total} GT anomalies  |  [{index}]  "
            f"anomaly=[{start}, {end}) len={seg_len}  "
            f"±{context_mult}×"
        )
    else:
        title = (
            f"{n_total} GT anomalies  |  [{index}]  "
            f"anomaly=[{start}, {end}) len={seg_len}  "
            f"±{context_mult}×  thr={threshold} (no PA)  "
            f"TP={n_tp} FP={n_fp} FN={n_fn}"
        )
    fig.suptitle(title, fontsize=9, y=0.995)

    for i in range(n_dims):
        ax = axes[i]
        if start > left:
            ax.axvspan(left, start, color=green, alpha=0.85, lw=0, zorder=0)
        if right > end:
            ax.axvspan(end, right, color=green, alpha=0.85, lw=0, zorder=0)
        for a, b in _segments(y_win):
            ax.axvspan(left + a, left + b, color=pink, alpha=0.9, lw=0, zorder=1)
        ax.axvline(start, color="#c62828", lw=0.8, ls=":", zorder=3)
        ax.axvline(end, color="#c62828", lw=0.8, ls=":", zorder=3)

        series = x_win[:, i].astype(float)
        ax.plot(xs, series, color=line_c, lw=1.0, zorder=2)

        if not gt_only and pred_win is not None:
            if n_tp:
                ax.scatter(
                    xs[tp_mask],
                    series[tp_mask],
                    s=10,
                    c=red,
                    marker="o",
                    linewidths=0,
                    zorder=5,
                    label="TP",
                )
            if n_fp:
                ax.scatter(
                    xs[fp_mask],
                    series[fp_mask],
                    s=14,
                    c=red,
                    marker="^",
                    linewidths=0,
                    zorder=5,
                    label="FP",
                )
            if n_fn:
                ax.scatter(
                    xs[fn_mask],
                    series[fn_mask],
                    s=12,
                    facecolors="none",
                    edgecolors=fn_c,
                    marker="o",
                    linewidths=0.9,
                    zorder=5,
                    label="FN",
                )

        lo, hi = float(series.min()), float(series.max())
        if hi - lo < 1e-12:
            ax.set_ylim(lo - 0.1, hi + 0.1)
        else:
            pad_y = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad_y, hi + pad_y)
        ax.set_ylabel(f"m{i}", fontsize=8, rotation=0, labelpad=16, va="center")
        ax.tick_params(labelbottom=(i == n_dims - 1), labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    axes[-1].set_xlabel("timestamp", fontsize=9)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=green, edgecolor="none", label=f"normal (±{context_mult}×)"),
        Patch(facecolor=pink, edgecolor="none", label="GT anomaly"),
        Line2D([0], [0], color=line_c, lw=1.2, label="metric"),
    ]
    if not gt_only:
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=red,
                    markersize=6,
                    linestyle="None",
                    label="TP (pred∩GT)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor=red,
                    markersize=7,
                    linestyle="None",
                    label="FP (pred∩¬GT)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=fn_c,
                    markerfacecolor="w",
                    markersize=6,
                    linestyle="None",
                    label="FN (GT∩¬pred)",
                ),
            ]
        )
    axes[0].legend(
        handles=handles,
        loc="upper right",
        fontsize=6.5,
        framealpha=0.92,
        ncol=3 if not gt_only else 1,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = args.out_dir or ("viz_gt" if args.gt_only else "viz_pred")

    if args.gt_only:
        x, y = _load_series_for_gt(args)
        pred = None
        thr = None
        thr_src = score_path = None
        exp = "gt"
        out_dir = os.path.join(out_root, args.dataset)
        prefix = "gt"
        header = f" GT ANOMALY ONLY  ({args.dataset})"
        run_dir = None
    else:
        run_dir = _resolve_run_dir(args)
        score, pred, thr, thr_src, score_path = _load_pred(run_dir, args.threshold)
        x, y = _load_series_for_pred(args, run_dir, n_score=len(score))
        # score 길이에 맞춤
        n = min(len(x), len(y), len(score), len(pred))
        x, y, score, pred = x[:n], y[:n], score[:n], pred[:n]
        assert len(x) == len(y) == len(pred) == len(score)
        exp = args.exp_name or os.path.basename(os.path.abspath(run_dir))
        out_dir = os.path.join(out_root, args.dataset, exp)
        prefix = "pred"
        header = f" GT ANOMALY + MODEL PRED  ({args.dataset} / {exp})"

    segs = _segments(y)
    if args.max_segments is not None:
        segs = segs[: args.max_segments]
    n_total = len(segs)
    os.makedirs(out_dir, exist_ok=True)

    print()
    print("=" * 64)
    print(header)
    print("=" * 64)
    if not args.gt_only:
        print(f"  experiment           : {exp}")
        print(f"  run_dir              : {run_dir}")
        print(f"  test_score           : {score_path}")
        print(f"  threshold            : {thr}  ({thr_src})")
        print("  point adjustment     : False")
        print("  score rule           : score > threshold → anomaly")
    print(f"  GT anomaly segments  : {n_total}")
    print(f"  GT anomaly points    : {int(y.sum())} / {len(y)}")
    print(f"  channels             : {1 if x.ndim == 1 else x.shape[1]}"
          + (f" (plot max_dims={args.max_dims})" if args.max_dims else ""))
    print(f"  context multiplier   : ±{args.context_mult}×")
    print(f"  output dir           : {out_dir}")
    if not args.gt_only:
        tp_all = int((y & pred).sum())
        fp_all = int(((~y) & pred).sum())
        fn_all = int((y & (~pred)).sum())
        tn_all = int(((~y) & (~pred)).sum())
        print("-" * 64)
        print(f"  TP points (no PA)    : {tp_all}")
        print(f"  FP points (no PA)    : {fp_all}")
        print(f"  FN points (no PA)    : {fn_all}")
        print(f"  TN points (no PA)    : {tn_all}")
    print("-" * 64)
    show_n = min(50, n_total)
    for i, (a, b) in enumerate(segs[:show_n]):
        print(f"  [{i:4d}]  start={a:8d}  end={b:8d}  len={b - a:6d}")
    if n_total > show_n:
        print(f"  ... ({n_total - show_n} more)")
    print("=" * 64)
    print(f"\nSaving {n_total} figures...")

    for i, (start, end) in enumerate(segs):
        out_path = os.path.join(out_dir, f"{prefix}_{i:04d}.png")
        plot_gt_segment(
            x,
            y,
            pred,
            start,
            end,
            i,
            n_total,
            args.context_mult,
            thr,
            out_path,
            gt_only=args.gt_only,
            max_dims=args.max_dims,
        )
        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            print(f"  [{i + 1}/{n_total}] {out_path}")

    print(f"\nDone. {n_total} figures saved under {out_dir}/")


if __name__ == "__main__":
    main()
