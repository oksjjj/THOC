# THOC: Temporal Hierarchical One-Class Network

NeurIPS 2020 논문 [*Timeseries Anomaly Detection using Temporal Hierarchical One-Class Network*](https://proceedings.neurips.cc/paper/2020/file/97e401a02082021fd24957f852e0e475-Paper.pdf)을 기반으로 한 PyTorch 구현입니다.

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

논문 §4.1 프로토콜 (기본 활성화):
- train: train 시계열 앞 70%
- validation: train 시계열 뒤 30% holdout (F1·threshold·HP 선택)
- test: test 시계열 **전체** (최종 평가)

```bash
# 기본 학습 (validation F1-PA 기준 checkpoint)
python main.py --dataset SMAP --epochs 30

# lr, λ_orth, λ_TSS 그리드 서치 → 최적 조합으로 재학습
python main.py --dataset SMAP --tune --tune_epochs 15 --epochs 30

# 하이퍼파라미터 직접 지정
python main.py --dataset SMAP --lr 5e-4 --lambda_orth 0.1 --lambda_tss 0.1 --epochs 30
```

### SMD (Server Machine Dataset)

```bash
python main.py --dataset SMD --epochs 30
```

`data/SMD/` 아래 npy 파일 사용:
- `SMD_train.npy`
- `SMD_test.npy`
- `SMD_test_label.npy`

## 로깅

학습 시 자동으로 다음 파일이 생성됩니다.

| 경로 | 내용 |
|------|------|
| `logs/{exp_name}.log` | 콘솔 + 파일 로그 |
| `outputs/{exp_name}/train_history.json` | epoch별 손실 기록 |
| `outputs/{exp_name}/results.json` | 평가 지표 (F1, F1-PA, AUC 등) |
| `checkpoints/{exp_name}/best.pt` | 최적 모델 체크포인트 |

로그 관련 옵션:

```bash
python main.py --dataset SMAP --log_level DEBUG --log_freq 5 --exp_name my_smap_run
```

## 데이터 준비

### 포함된 데이터

```
data/
├── NeurIPS-TS/     # CSV (단변량/다변량)
├── SMAP/           # SMAP_train.npy, SMAP_test.npy, SMAP_test_label.npy
└── SMD/            # SMD_train.npy, SMD_test.npy, SMD_test_label.npy
```

### 원본 데이터 전처리

원본 txt/npy 파일이 있는 경우:

```bash
# SMD
python scripts/preprocess_data.py --dataset SMD \
  --raw_dir /path/to/ServerMachineDataset \
  --output_dir data/SMD

# SMAP
python scripts/preprocess_data.py --dataset SMAP \
  --raw_dir /path/to/raw_smap \
  --output_dir data/SMAP
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
| `--log_dir` | ./logs | 로그 디렉터리 |
| `--output_dir` | ./outputs | 결과 JSON 저장 경로 |

## 평가 지표

- **F1 / Precision / Recall**: Youden's J statistic으로 threshold 선택
- **F1-PA**: Point-Adjust 평가 (SMAP/SMD 벤치마크 표준)
- **AUC**: ROC-AUC

## 프로젝트 구조

```
THOC_gpt/
├── main.py
├── requirements.txt
├── scripts/
│   └── preprocess_data.py
├── thoc/
│   ├── model.py
│   ├── data.py
│   ├── metrics.py
│   ├── logger.py
│   └── trainer.py
├── data/
│   ├── NeurIPS-TS/
│   ├── SMAP/
│   └── SMD/
├── logs/
├── outputs/
└── checkpoints/
```

## Git

Remote: [github.com/oksjjj/THOC](https://github.com/oksjjj/THOC)

`data/`, `checkpoints/`, `logs/`, `outputs/`, `paper/`는 `.gitignore`에 포함됩니다.

```bash
git init
git remote add origin https://github.com/oksjjj/THOC.git
git add .
git commit -m "Initial commit: THOC PyTorch implementation"
git branch -M main
git push -u origin main
```
