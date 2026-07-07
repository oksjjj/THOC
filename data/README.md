# Data

This directory is not tracked in git (large files). After cloning, place datasets here:

```
data/
├── NeurIPS-TS/     # nts_uni_*.csv, nts_mul_*.csv
├── SMAP/           # SMAP_train.npy, SMAP_test.npy, SMAP_test_label.npy
└── SMD/            # SMD_train.npy, SMD_test.npy, SMD_test_label.npy
```

Preprocess from raw sources:

```bash
python scripts/preprocess_data.py --dataset SMAP --raw_dir /path/to/raw --output_dir data/SMAP
python scripts/preprocess_data.py --dataset SMD --raw_dir /path/to/ServerMachineDataset --output_dir data/SMD
```

NeurIPS-TS CSV files: [NeurIPS-TS repository](https://github.com/elisejiuqian/NeurIPS-TS).
