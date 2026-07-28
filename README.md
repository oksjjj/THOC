# THOC: Temporal Hierarchical One-Class Network

NeurIPS 2020 논문 [*Timeseries Anomaly Detection using Temporal Hierarchical One-Class Network*](https://proceedings.neurips.cc/paper/2020/file/97e401a02082021fd24957f852e0e475-Paper.pdf)을 기반으로 한 PyTorch 구현입니다.

출력 디렉터리 레이아웃은 OmniAnomaly와 동일합니다: `{model|result|log}/{dataset}/{exp}/`.

## 구조

- **Dilated RNN**: 다중 스케일 시간 특징 추출
- **계층적 클러스터링**: 각 해상도에서 soft assignment로 특징 융합
- **MVDD (Multiscale Vector Data Description)**: 다중 초구면 기반 one-class 목적함수
- **L_orth**: 클러스터 중심 직교성 손실
- **L_TSS**: 다중 스텝 ahead 예측 기반 temporal self-supervision 손실

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

### NeurIPS-TS

```bash
python main.py --dataset NeurIPS-TS-UNI --epochs 30
python main.py --dataset NeurIPS-TS-MUL --epochs 30
```

### SMAP (NASA)

논문 설정: `window_size=100`, `test_stride=100` (기본값으로 적용됨)

SMAP/SMD 분할 (기본):
- train: train 시계열 **전체** (one-class 학습)
- validation: test 시계열 **앞 15%** (F1·threshold·HP 선택)
- test: test 시계열 **나머지 85%** (최종 평가)

```bash
python main.py --dataset SMAP --epochs 30 --device mps
python main.py --dataset SMAP --val_ratio 0.15 --tune --tune_epochs 5 --epochs 30 --device mps
```

### SMD (Server Machine Dataset)

```bash
# 전체 concat
python main.py --dataset SMD --epochs 30

# 머신별 (OmniAnomaly 와 동일)
python main.py --dataset machine-1-1 --epochs 30
python main.py --all_smd_machines --epochs 30
```

## 로깅 / 산출물

학습 시 자동으로 다음이 생성됩니다 (OmniAnomaly 동일 중첩).

| 경로 | 내용 |
|------|------|
| `log/{dataset}/{exp}/*.log` | 콘솔 + 파일 로그 |
| `result/{dataset}/{exp}/train_history.json` | epoch별 손실 기록 |
| `result/{dataset}/{exp}/results.json` | 평가 지표 |
| `result/{dataset}/{exp}/roc_pr_*.png` | PA ROC/PR 곡선 |
| `model/{dataset}/{exp}/best.pt` | 최적 모델 체크포인트 |
| `viz_gt/{dataset}/` · `viz_pred/{dataset}/{exp}/` | GT / pred 시각화 |

```bash
python main.py --dataset SMAP --log_level DEBUG --log_freq 5 --exp_name my_smap_run
# → model/SMAP/my_smap_run/ , result/SMAP/my_smap_run/ , log/SMAP/my_smap_run/
```

## 데이터 준비

```
data/
├── NeurIPS-TS/
├── SMAP/
├── SMD/
└── raw/            # 선택 (원본)
```

자세한 내용은 `data/README.md` 참고.

```bash
python scripts/preprocess_data.py --dataset SMD \
  --raw_dir /path/to/ServerMachineDataset --output_dir data/SMD
python scripts/preprocess_data.py --dataset SMAP \
  --raw_dir /path/to/raw_smap --output_dir data/SMAP
```

## 주요 하이퍼파라미터

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--window_size` | 데이터셋별 자동 | 슬라이딩 윈도우 크기 |
| `--stride` | 데이터셋별 자동 | 학습 stride |
| `--test_stride` | 데이터셋별 자동 | 평가 stride (SMAP: 100) |
| `--hidden_dim` | 128 | DRNN hidden dimension |
| `--lr` | 1e-3 | 학습률 |
| `--lambda_orth` | 1.0 | 직교 손실 가중치 |
| `--lambda_tss` | 1.0 | TSS 손실 가중치 |
| `--log_dir` | log | 로그 루트 |
| `--result_dir` | result | 결과 루트 (`--output_dir` 별칭) |
| `--save_dir` | model | 체크포인트 루트 (`--checkpoint_dir` 별칭) |
| `--exp_name` | 자동 | `{dataset}` 아래 실험 폴더명 |

## 평가 지표

- **F1 / Precision / Recall**: Youden's J statistic으로 threshold 선택
- **F1-PA**: Point-Adjust 평가 (SMAP/SMD 벤치마크 표준)
- **AUC**: ROC-AUC / AUPRC (PA)

## 프로젝트 구조

```
THOC/
├── main.py
├── requirements.txt
├── thoc/                 # 모델 · 학습 · 평가 패키지
├── scripts/
│   ├── preprocess_data.py
│   └── viz_gt_anomalies.py
├── data/
│   ├── NeurIPS-TS/
│   ├── SMAP/
│   ├── SMD/
│   └── raw/
├── model/ · result/ · log/
└── viz_gt/ · viz_pred/
```

## Git

Remote: [github.com/oksjjj/THOC](https://github.com/oksjjj/THOC)

`data/`, `model/`, `log/`, `result/`, `viz_gt/`, `viz_pred/`는 `.gitignore`에 포함됩니다.
