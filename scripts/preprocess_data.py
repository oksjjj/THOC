#!/usr/bin/env python3
"""
원본 SMD / SMAP / MSL 데이터를 학습용 pickle 파일로 변환하는 전처리 스크립트.

THOC 학습 코드(`thoc/data.py`)는 `data/SMAP/*.npy` 형태를 직접 읽을 수 있지만,
OmniAnomaly [28] 계열 원본 레이아웃(txt, 개별 npy)을 쓰는 경우 이 스크립트로
한 번에 concat + pickle 저장을 수행한다.

────────────────────────────────────────────────────────────
원본 디렉터리 구조
────────────────────────────────────────────────────────────

SMD (OmniAnomaly):
  ServerMachineDataset/
    train/machine-1-1.txt          ← 정상 구간
    test/machine-1-1.txt           ← 테스트 시계열
    test_label/machine-1-1.txt     ← 시점별 0/1 라벨

SMAP / MSL (NASA):
  data/
    train/A-1.npy                  ← 채널별 정상 train
    test/A-1.npy                   ← 채널별 test
    labeled_anomalies.csv          ← 이상 구간 메타정보

────────────────────────────────────────────────────────────
출력 예시
────────────────────────────────────────────────────────────

SMD:
  SMD_train.pkl, SMD_test.pkl, SMD_test_label.pkl
  (+ machine별 개별 pkl)

SMAP:
  SMAP_train.pkl, SMAP_test.pkl, SMAP_test_label.pkl

사용 예:
  python scripts/preprocess_data.py --dataset SMAP --raw_dir data --output_dir data/SMAP
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import pickle

import numpy as np


def save_pickle(path: str, data: np.ndarray) -> None:
    """numpy 배열을 pickle 파일로 저장하고 shape 를 출력한다."""
    # output_dir 이 아직 없으면 생성
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(data, file)
    print(f"Saved {path} | shape={data.shape}")


def preprocess_smd(raw_dir: str, output_dir: str) -> None:
    """
    SMD (Server Machine Dataset) 전처리.

    28개 machine × (train / test / label) txt 를 읽어
    machine 단위 pkl + 전체 concat pkl 을 만든다.
    """
    # OmniAnomaly 원본은 split 별 하위 폴더 구조
    for split in ("train", "test", "test_label"):
        split_dir = os.path.join(raw_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing directory: {split_dir}")

    # concat 결과를 담을 리스트 — machine 순서대로 append
    train_parts, test_parts, label_parts = [], [], []

    # train/ 아래 machine-1-1.txt, machine-1-2.txt, ... 정렬 목록
    filenames = sorted(
        name for name in os.listdir(os.path.join(raw_dir, "train")) if name.endswith(".txt")
    )

    for filename in filenames:
        machine_id = filename.removesuffix(".txt")  # "machine-1-1"

        # 한 machine 에 대해 train / test / label 세 파일 처리
        for split, container in (
            ("train", train_parts),
            ("test", test_parts),
            ("test_label", label_parts),
        ):
            path = os.path.join(raw_dir, split, filename)
            # CSV 형식 txt → float32 배열 (시점 × 38차원)
            data = np.genfromtxt(path, dtype=np.float32, delimiter=",")
            # machine 단위 개별 pkl (디버깅·개별 분석용)
            save_pickle(os.path.join(output_dir, f"{machine_id}_{split}.pkl"), data)
            container.append(data)

    # 28개 machine 을 순서대로 이어 붙인 전체 배열
    # thoc/data.py 의 SMD_train.npy 와 동일한 concat 형태
    save_pickle(os.path.join(output_dir, "SMD_train.pkl"), np.asarray(train_parts))
    save_pickle(os.path.join(output_dir, "SMD_test.pkl"), np.asarray(test_parts))
    save_pickle(os.path.join(output_dir, "SMD_test_label.pkl"), np.asarray(label_parts))


def preprocess_smap_msl(dataset: str, raw_dir: str, output_dir: str) -> None:
    """
    SMAP / MSL 전처리.

    labeled_anomalies.csv 에서 채널 목록·이상 구간을 읽고,
    채널별 npy 를 concat 해 하나의 긴 시계열로 만든다.
    """
    csv_path = os.path.join(raw_dir, "labeled_anomalies.csv")
    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))[1:]  # 헤더 행 제외

    # dataset 컬럼이 "SMAP" 또는 "MSL" 인 행만 선택
    # P-2 채널은 논문/벤치마크에서 제외
    data_info = sorted(
        [row for row in rows if row[1] == dataset and row[0] != "P-2"],
        key=lambda row: row[0],  # 채널 ID 알파벳 순 (A-1, A-2, ...)
    )

    # ── [1] 채널별 이상 라벨 생성 ──────────────────────────────────────
    labels: list[np.ndarray] = []
    for row in data_info:
        # row[2]: 이상 구간 리스트 문자열, 예) "[[100, 200], [500, 520]]"
        anomalies = ast.literal_eval(row[2])
        length = int(row[-1])  # 해당 채널 test 시계열 길이
        label = np.zeros(length, dtype=np.int64)  # 0=정상
        for start, end in anomalies:
            label[start : end + 1] = 1  # 이상 구간을 1로 표시
        labels.append(label)

    # ── [2] train / test 시계열 concat ─────────────────────────────────
    for split in ("train", "test"):
        chunks = [
            np.load(os.path.join(raw_dir, split, f"{row[0]}.npy"))
            for row in data_info  # 채널 순서 = data_info 정렬 순서
        ]
        # np.asarray(chunks) → (총시점, 25) 형태의 하나의 긴 시계열
        save_pickle(os.path.join(output_dir, f"{dataset}_{split}.pkl"), np.asarray(chunks))

    # ── [3] 채널별 라벨을 이어 붙여 전체 test 라벨 생성 ───────────────
    save_pickle(
        os.path.join(output_dir, f"{dataset}_test_label.pkl"),
        np.asarray(np.concatenate(labels)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess anomaly detection datasets")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["SMD", "SMAP", "MSL"],
        help="전처리할 데이터셋 이름",
    )
    parser.add_argument(
        "--raw_dir",
        required=True,
        help="원본 데이터 루트 (SMD: ServerMachineDataset/, SMAP: data/)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="pickle 출력 디렉터리",
    )
    args = parser.parse_args()

    if args.dataset == "SMD":
        preprocess_smd(args.raw_dir, args.output_dir)
    else:
        preprocess_smap_msl(args.dataset, args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
