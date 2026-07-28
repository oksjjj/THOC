"""
시계열 이상 탐지 데이터 로더.

논문 §4.1 전처리 방식:
  - 슬라이딩 윈도우로 고정 길이 시퀀스 생성
  - SMAP/MSL: window=100, stride=100 (평가 시)
  - 기타: window=80~100, stride=1

학습은 라벨 없이(normal 데이터만) one-class 방식으로 진행한다 (§3).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset

DatasetName = Literal["NeurIPS-TS-UNI", "NeurIPS-TS-MUL", "SMAP", "SMD"]
ScalerName = Literal["std", "minmax"]

# SMAP / SMD(전체·machine-*): train 전체 + test 앞 val_ratio → validation
VAL_SPLIT_DATASETS: frozenset[str] = frozenset({"SMAP", "SMD"})
DEFAULT_VAL_RATIO: dict[str, float] = {"SMAP": 0.15, "SMD": 0.15}
NEURIPS_VAL_RATIO: float = 0.3
SMD_MACHINE_RE = re.compile(r"^machine-\d+-\d+$")


def is_smd_machine(dataset: str) -> bool:
    """OmniAnomaly 스타일 SMD entity 이름 (예: machine-1-1)."""
    return bool(SMD_MACHINE_RE.match(str(dataset)))


def uses_test_front_val(dataset: str) -> bool:
    """test 시계열 앞부분을 validation 으로 쓰는 데이터셋인지."""
    return dataset in VAL_SPLIT_DATASETS or is_smd_machine(dataset)


def resolve_val_ratio(dataset: str, val_ratio: float | None) -> float:
    """데이터셋별 validation 비율 기본값."""
    if val_ratio is not None:
        return val_ratio
    if uses_test_front_val(dataset):
        return DEFAULT_VAL_RATIO.get("SMD" if is_smd_machine(dataset) else dataset, 0.15)
    return NEURIPS_VAL_RATIO


@dataclass(frozen=True)
class DatasetConfig:
    data_dir: str
    window_size: int
    stride: int
    test_stride: int | None = None


# 논문 Table 1 및 §4.1 기반 기본 하이퍼파라미터
DATASET_DEFAULTS: dict[str, DatasetConfig] = {
    "NeurIPS-TS-UNI": DatasetConfig("data/NeurIPS-TS", window_size=64, stride=1),
    "NeurIPS-TS-MUL": DatasetConfig("data/NeurIPS-TS", window_size=64, stride=1),
    "SMAP": DatasetConfig("data/SMAP", window_size=100, stride=1, test_stride=100),
    "SMD": DatasetConfig("data/SMD", window_size=100, stride=1),
}


def get_dataset_defaults(dataset: str) -> DatasetConfig:
    if is_smd_machine(dataset):
        return DATASET_DEFAULTS["SMD"]
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Choose from {list(DATASET_DEFAULTS)} or machine-{{g}}-{{id}}"
        )
    return DATASET_DEFAULTS[dataset]


def _load_array(path_stem: str) -> np.ndarray:
    """``.npy`` 우선, 없으면 ``.pkl`` 로드."""
    npy_path = f"{path_stem}.npy"
    pkl_path = f"{path_stem}.pkl"
    if os.path.isfile(npy_path):
        return np.load(npy_path)
    if os.path.isfile(pkl_path):
        import pickle

        with open(pkl_path, "rb") as file:
            data = pickle.load(file)
        return np.asarray(data)
    raise FileNotFoundError(f"Missing data file: {npy_path} or {pkl_path}")


def load_neurips_ts(
    data_dir: str,
    dataset: Literal["NeurIPS-TS-UNI", "NeurIPS-TS-MUL"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dataset == "NeurIPS-TS-UNI":
        normal_path = os.path.join(data_dir, "nts_uni_normal.csv")
        abnormal_path = os.path.join(data_dir, "nts_uni_abnormal.csv")
    else:
        normal_path = os.path.join(data_dir, "nts_mul_normal.csv")
        abnormal_path = os.path.join(data_dir, "nts_mul_abnormal.csv")

    normal = pd.read_csv(normal_path)
    abnormal = pd.read_csv(abnormal_path)

    train_x = normal.iloc[:, :-1].to_numpy(dtype=np.float32)
    train_y = normal.iloc[:, -1].to_numpy(dtype=np.int64)
    test_x = abnormal.iloc[:, :-1].to_numpy(dtype=np.float32)
    test_y = abnormal.iloc[:, -1].to_numpy(dtype=np.int64)
    return train_x, train_y, test_x, test_y


def load_smap(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    NASA SMAP 데이터셋 (논문 Table 1: dim=25, length=100).

    학습 데이터는 라벨 없이 사용 (§3 unsupervised setting).
    train_y는 모두 0으로 설정 — 학습 시 라벨을 쓰지 않음.
    """
    train_x = _load_array(os.path.join(data_dir, "SMAP_train")).astype(np.float32)
    test_x = _load_array(os.path.join(data_dir, "SMAP_test")).astype(np.float32)
    test_y = _load_array(os.path.join(data_dir, "SMAP_test_label")).astype(np.int64)
    train_y = np.zeros(train_x.shape[0], dtype=np.int64)
    return train_x, train_y, test_x, test_y


def load_smd(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Server Machine Dataset (SMD) — 38차원, 28개 machine concatenated.

    data/SMD/ 아래 npy(또는 pkl) 파일 사용:
      - SMD_train.npy
      - SMD_test.npy
      - SMD_test_label.npy

    각 machine은 전반부=train, 후반부=test로 분할되어 있다.
    """
    train_x = _load_array(os.path.join(data_dir, "SMD_train")).astype(np.float32)
    test_x = _load_array(os.path.join(data_dir, "SMD_test")).astype(np.float32)
    test_y = _load_array(os.path.join(data_dir, "SMD_test_label")).astype(np.int64)
    if train_x.ndim != 2 or test_x.ndim != 2:
        raise ValueError(
            "SMD arrays must be 2D (T, 38). Re-run scripts/preprocess_data.py "
            "to concatenate machine parts."
        )
    train_y = np.zeros(train_x.shape[0], dtype=np.int64)
    return train_x, train_y, test_x, test_y


def list_smd_machines(data_dir: str = "data/SMD") -> list[str]:
    """data_dir 에 있는 machine-* entity 목록 (정렬)."""
    if not os.path.isdir(data_dir):
        return []
    machines: set[str] = set()
    for name in os.listdir(data_dir):
        stem = name
        for suffix in ("_train.npy", "_train.pkl", "_test.npy", "_test.pkl"):
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                break
        if is_smd_machine(stem):
            machines.add(stem)
    return sorted(machines, key=lambda m: tuple(int(p) for p in m.split("-")[1:]))


def load_smd_machine(
    data_dir: str,
    machine_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    SMD 단일 machine (OmniAnomaly ``--dataset machine-1-1`` 대응).

    기대 파일:
      {data_dir}/{machine_id}_train.npy|.pkl
      {data_dir}/{machine_id}_test.npy|.pkl
      {data_dir}/{machine_id}_test_label.npy|.pkl
    """
    if not is_smd_machine(machine_id):
        raise ValueError(f"Invalid SMD machine id: {machine_id}")

    train_x = _load_array(os.path.join(data_dir, f"{machine_id}_train")).astype(np.float32)
    test_x = _load_array(os.path.join(data_dir, f"{machine_id}_test")).astype(np.float32)
    test_y = _load_array(os.path.join(data_dir, f"{machine_id}_test_label")).astype(np.int64)
    if train_x.ndim == 1:
        train_x = train_x.reshape(-1, 1)
    if test_x.ndim == 1:
        test_x = test_x.reshape(-1, 1)
    if train_x.ndim != 2 or test_x.ndim != 2:
        raise ValueError(
            f"{machine_id} arrays must be 2D (T, C). Got train={train_x.shape} test={test_x.shape}"
        )
    train_y = np.zeros(train_x.shape[0], dtype=np.int64)
    return train_x, train_y, test_x, test_y


LOADERS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {
    "NeurIPS-TS-UNI": load_neurips_ts,
    "NeurIPS-TS-MUL": load_neurips_ts,
    "SMAP": load_smap,
    "SMD": load_smd,
}


@dataclass(frozen=True)
class DatasetSplits:
    """
    논문 Table 1 프로토콜에 맞춘 train / validation / test 분할.

    SMAP·SMD:
      - train: train 시계열 전체 (one-class 학습)
      - val:   test 시계열 앞 val_ratio (F1·threshold·HP 선택, 라벨 있음)
      - test:  test 시계열 나머지 (최종 평가)

    NeurIPS-TS:
      - train: normal 전체
      - val:   abnormal 앞 30%
      - test:  abnormal 뒤 70%
    """

    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    val_x: np.ndarray | None = None
    val_y: np.ndarray | None = None


def temporal_split(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    val_position: Literal["start", "end"] = "end",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    시계열을 시간 순서를 유지한 채 두 구간으로 분할.

    Returns:
        (first_x, first_y, second_x, second_y)

    val_position="end":   [train | val]   — first=앞 (1-r), second=뒤 (r)
    val_position="start": [val | test]    — first=앞 (r),   second=뒤 (1-r)
      SMAP/SMD·NeurIPS-TS: test 시계열 앞 val_ratio → validation
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    if val_position == "end":
        split_idx = int(len(x) * (1.0 - val_ratio))
        return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]

    # start: 앞쪽 val_ratio 가 validation
    split_idx = int(len(x) * val_ratio)
    return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]


def make_dataset_splits(
    dataset: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    val_ratio: float = 0.3,
) -> DatasetSplits:
    """
    데이터셋별 train / validation / test 분할.

    SMAP·SMD: train 전체, test 앞 val_ratio → val, test 나머지 → test.
    NeurIPS-TS: normal 전체 train, abnormal 앞 val_ratio → val, 나머지 → test.
    """
    if val_ratio <= 0:
        return DatasetSplits(train_x, train_y, test_x, test_y)

    if uses_test_front_val(dataset):
        val_x, val_y, te_x, te_y = temporal_split(
            test_x, test_y, val_ratio=val_ratio, val_position="start"
        )
        return DatasetSplits(train_x, train_y, te_x, te_y, val_x=val_x, val_y=val_y)

    if dataset in ("NeurIPS-TS-UNI", "NeurIPS-TS-MUL"):
        val_x, val_y, te_x, te_y = temporal_split(
            test_x, test_y, val_ratio=val_ratio, val_position="start"
        )
        return DatasetSplits(train_x, train_y, te_x, te_y, val_x=val_x, val_y=val_y)

    return DatasetSplits(train_x, train_y, test_x, test_y)


def load_dataset(
    dataset: str,
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if is_smd_machine(dataset):
        return load_smd_machine(data_dir, dataset)

    loader = LOADERS.get(dataset)
    if loader is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    if dataset in ("NeurIPS-TS-UNI", "NeurIPS-TS-MUL"):
        return loader(data_dir, dataset)  # type: ignore[arg-type]
    return loader(data_dir)


class TimeSeriesAnomalyDataset(Dataset):
    """
    슬라이딩 윈도우 Dataset.

    윈도우 i의 입력:  x[i·stride : i·stride + window_size]
    윈도우 i의 라벨:  마지막 시점 y[i·stride + window_size - 1]
                      (또는 window_label=True 시 구간 내 any anomaly)
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window_size: int,
        stride: int,
        scaler: preprocessing.StandardScaler | preprocessing.MinMaxScaler,
        fit_scaler: bool,
        window_label: bool = False,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.window_label = window_label

        # 슬라이딩 윈도우로 만들 수 있는 개수 계산
        #   윈도우 i의 시작 위치 = i * stride
        #   마지막 윈도우 시작 = (usable_len - 1) * stride
        #   → (N - window_size) // stride + 1
        # 예) N=1000, window=64, stride=1  → 937개 윈도우
        # 예) N=1000, window=100, stride=100 → 10개 윈도우
        usable_len = (x.shape[0] - window_size) // stride + 1

        # 시계열을 window_size 단위로 딱 맞게 자르기 위한 길이
        #   trim_len = usable_len * window_size
        # stride=window_size (SMAP 등)일 때 효과적:
        #   N=1050, window=100, stride=100 → usable_len=10, trim_len=1000
        #   → 뒤 50시점은 윈도우에 안 들어가므로 잘라냄
        # stride=1일 때는 trim_len이 N보다 커서 실질적으로 자르지 않음
        trim_len = usable_len * window_size
        x = x[:trim_len]  # 입력 시계열을 trim_len까지만 사용
        y = y[:trim_len]  # 라벨도 동일 길이로 맞춤 (시점별 1:1 대응)

        # train에서 fit, test에서 transform — 데이터 누수 방지
        if fit_scaler:
            self.x = scaler.fit_transform(x).astype(np.float32)
        else:
            self.x = scaler.transform(x).astype(np.float32)
        self.y = y
        self.scaler = scaler
        self.length = usable_len

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        DataLoader가 호출하는 함수. idx번째 슬라이딩 윈도우 1개를 반환.

        예) window_size=4, stride=2, 시계열 x = [0,1,2,3,4,5,6,7,8,9]
            idx=0 → x[0:4] = [0,1,2,3]
            idx=1 → x[2:6] = [2,3,4,5]
            idx=2 → x[4:8] = [4,5,6,7]
        """
        # 윈도우 시작/끝 인덱스
        start = idx * self.stride              # i번째 윈도우 시작 = i * stride
        end = start + self.window_size         # 끝 = 시작 + window_size (미포함)

        # 입력: (window_size, channels) 텐서
        window_x = torch.from_numpy(self.x[start:end])
        # 이 윈도우 구간의 라벨 (시점별 0/1)
        window_y = self.y[start:end]

        if self.window_label:
            # 윈도우 안에 이상(1)이 하나라도 있으면 이상 윈도우
            label = int(np.any(window_y == 1))
        else:
            # 윈도우 마지막 시점의 라벨을 대표값으로 사용
            # (THOC는 이 시점의 AnomalyScore와 비교)
            label = int(window_y[-1])

        return window_x, label


def build_dataloaders(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    window_size: int,
    stride: int,
    batch_size: int,
    eval_batch_size: int,
    scaler_name: ScalerName = "std",
    test_stride: int | None = None,
    window_label: bool = False,
    val_x: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
) -> tuple[
    TimeSeriesAnomalyDataset,
    DataLoader,
    TimeSeriesAnomalyDataset | None,
    DataLoader | None,
    TimeSeriesAnomalyDataset,
    DataLoader,
]:
    """
    train/test 시계열 → PyTorch DataLoader 변환.

    흐름:
      1. scaler 생성 (표준화 or MinMax)
      2. train Dataset: scaler.fit_transform (평균/분산 학습)
      3. test  Dataset: scaler.transform     (train 통계로 변환, 누수 방지)
      4. DataLoader로 배치 단위 반복 가능하게 래핑
    """
    # 표준화 방식 선택
    #   std    → (x - mean) / std  (각 채널별)
    #   minmax → (x - min) / (max - min)
    if scaler_name == "std":
        scaler = preprocessing.StandardScaler()
    else:
        scaler = preprocessing.MinMaxScaler()

    # ── Train Dataset ──
    # fit_scaler=True: train 데이터로 mean/std를 학습하고 변환
    # stride: 윈도우 이동 간격 (학습 시 보통 1)
    train_dataset = TimeSeriesAnomalyDataset(
        train_x,
        train_y,
        window_size=window_size,
        stride=stride,
        scaler=scaler,
        fit_scaler=True,
        window_label=window_label,
    )

    # ── Test Dataset ──
    # fit_scaler=False: train에서 학습한 scaler로만 변환 (test 정보 유출 방지)
    # test_stride: 평가용 stride (SMAP은 100 → non-overlapping 윈도우)
    test_dataset = TimeSeriesAnomalyDataset(
        test_x,
        test_y,
        window_size=window_size,
        stride=test_stride if test_stride is not None else stride,
        scaler=scaler,
        fit_scaler=False,
        window_label=window_label,
    )

    # shuffle=True: 매 epoch마다 윈도우 순서 섞음 (학습 안정화)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    # shuffle=False: 평가 시 시간 순서 유지 (점수 복원에 필요)
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

    val_dataset: TimeSeriesAnomalyDataset | None = None
    val_loader: DataLoader | None = None
    if val_x is not None and val_y is not None:
        val_dataset = TimeSeriesAnomalyDataset(
            val_x,
            val_y,
            window_size=window_size,
            stride=test_stride if test_stride is not None else stride,
            scaler=scaler,
            fit_scaler=False,
            window_label=window_label,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
        )

    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader
