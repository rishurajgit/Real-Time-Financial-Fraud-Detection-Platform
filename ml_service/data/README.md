# IEEE-CIS Fraud Detection Dataset Setup

SentinelRisk uses the official **IEEE-CIS Fraud Detection** dataset from Kaggle as its primary training and evaluation dataset. Raw dataset files are intentionally excluded from Git due to licensing and size (~1.3 GB unzipped).

SentinelRisk **never fabricates synthetic dataset records** when the CSV files are missing. All EDA and model training commands validate the presence of these files first.

## Required Files

Place the extracted CSV files inside `ml_service/data/raw/`:

```text
ml_service/data/raw/
├── train_transaction.csv   (required for EDA & training)
├── train_identity.csv      (required for EDA & training)
├── test_transaction.csv    (optional; used for unlabeled scoring/replay)
└── test_identity.csv       (optional; merged with test_transaction.csv)
```

## Download Instructions

### Option A — Kaggle CLI (Recommended)

1. Accept the competition rules at:
   https://www.kaggle.com/competitions/ieee-fraud-detection/rules
2. Configure your Kaggle API token (`~/.kaggle/kaggle.json`).
3. Download and extract into `ml_service/data/raw/`:

```bash
mkdir -p ml_service/data/raw
kaggle competitions download -c ieee-fraud-detection -p ml_service/data/raw
unzip ml_service/data/raw/ieee-fraud-detection.zip -d ml_service/data/raw
rm ml_service/data/raw/ieee-fraud-detection.zip
```

### Option B — Manual Browser Download

1. Visit https://www.kaggle.com/competitions/ieee-fraud-detection/data
2. Download `ieee-fraud-detection.zip`.
3. Extract `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, and `test_identity.csv` directly into `ml_service/data/raw/`.

## Verification

Run the Phase 1 EDA pipeline to verify the files load and merge cleanly on `TransactionID`:

```bash
cd ml_service
python -m src.eda.analyze
```
