# Data

이 디렉터리는 git에 추적하지 않습니다 (대용량). 클론 후 아래처럼 준비하세요.

```
data/
├── NeurIPS-TS/                   # nts_uni_*.csv, nts_mul_*.csv
├── SMAP/                         # SMAP_{train,test,test_label}.{npy,pkl}
├── SMD/                          # SMD_* 및 machine-*_{train,test,test_label}
└── raw/                          # (선택) 원본 — OmniAnomaly 동일 레이아웃
    ├── nasa/                     # SMAP/MSL 원본
    └── ServerMachineDataset/     # SMD 원본
```

## 준비

```bash
# SMD
python scripts/preprocess_data.py --dataset SMD \
  --raw_dir data/raw/ServerMachineDataset --output_dir data/SMD

# SMAP
python scripts/preprocess_data.py --dataset SMAP \
  --raw_dir /path/to/raw --output_dir data/SMAP
```

NeurIPS-TS CSV: [NeurIPS-TS repository](https://github.com/elisejiuqian/NeurIPS-TS).
