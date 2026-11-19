"""IEEE-CIS Fraud Detection dataset loader, validator, and chronological splitter.

Supports official Kaggle competition CSV files:
  - train_transaction.csv
  - train_identity.csv
  - test_transaction.csv
  - test_identity.csv

Never fabricates synthetic rows if raw competition files are absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

TRAIN_TRANSACTION_FILE = "train_transaction.csv"
TRAIN_IDENTITY_FILE = "train_identity.csv"
TEST_TRANSACTION_FILE = "test_transaction.csv"
TEST_IDENTITY_FILE = "test_identity.csv"

REQUIRED_TRANSACTION_COLUMNS = ("TransactionID", "TransactionDT", "TransactionAmt")
TARGET_COLUMN = "isFraud"


class MissingDatasetError(FileNotFoundError):
    """Raised when required IEEE-CIS Kaggle CSV files are not present on disk."""


@dataclass(frozen=True)
class DatasetPaths:
    train_transaction: Path
    train_identity: Path
    test_transaction: Path
    test_identity: Path


def resolve_dataset_paths(data_dir: Path) -> DatasetPaths:
    """Resolve expected paths for all four IEEE-CIS CSV files."""
    base = Path(data_dir)
    return DatasetPaths(
        train_transaction=base / TRAIN_TRANSACTION_FILE,
        train_identity=base / TRAIN_IDENTITY_FILE,
        test_transaction=base / TEST_TRANSACTION_FILE,
        test_identity=base / TEST_IDENTITY_FILE,
    )


def validate_required_files(data_dir: Path, split: str = "train") -> tuple[Path, Path]:
    """Ensure required transaction and identity CSV files exist before reading."""
    paths = resolve_dataset_paths(data_dir)
    if split == "train":
        tx_path, id_path = paths.train_transaction, paths.train_identity
    elif split == "test":
        tx_path, id_path = paths.test_transaction, paths.test_identity
    else:
        raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'test'.")

    missing = [str(p) for p in (tx_path, id_path) if not p.is_file()]
    if missing:
        raise MissingDatasetError(
            "Required IEEE-CIS Fraud Detection files are missing:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nDownload the dataset from:\n"
            + "  https://www.kaggle.com/competitions/ieee-fraud-detection/data\n"
            + "and place the extracted CSV files in:\n"
            + f"  {Path(data_dir).resolve()}\n"
            + "See ml_service/data/README.md for full instructions."
        )
    return tx_path, id_path


def normalize_identity_columns(columns: list[str]) -> dict[str, str]:
    """Normalize IEEE-CIS test_identity.csv hyphenated columns (id-01 -> id_01).

    In the official Kaggle IEEE-CIS release, train_identity.csv uses underscores
    (id_01 .. id_38) whereas test_identity.csv uses hyphens (id-01 .. id-38).
    """
    rename_map: dict[str, str] = {}
    for col in columns:
        if col.startswith("id-"):
            rename_map[col] = col.replace("id-", "id_", 1)
    return rename_map


def load_ieee_cis_split(
    data_dir: Path,
    split: str = "train",
    nrows: int | None = None,
) -> "pd.DataFrame":
    """Load and left-join IEEE-CIS transaction and identity tables on TransactionID.

    Args:
        data_dir: Directory containing the extracted Kaggle CSV files.
        split: Either 'train' (contains isFraud target) or 'test'.
        nrows: Optional row limit for bounded-memory local experiments.

    Returns:
        Merged pandas DataFrame sorted chronologically by TransactionDT, TransactionID.
    """
    import pandas as pd

    tx_path, id_path = validate_required_files(data_dir, split=split)
    logger.info("Loading IEEE-CIS %s transactions from %s", split, tx_path)
    tx_df = pd.read_csv(tx_path, nrows=nrows, low_memory=False)

    missing_cols = [c for c in REQUIRED_TRANSACTION_COLUMNS if c not in tx_df.columns]
    if missing_cols:
        raise ValueError(f"{tx_path.name} is missing required columns: {missing_cols}")

    if split == "train":
        if TARGET_COLUMN not in tx_df.columns:
            raise ValueError(f"{tx_path.name} is missing target column '{TARGET_COLUMN}'.")
        if tx_df[TARGET_COLUMN].isna().any():
            raise ValueError(f"Target column '{TARGET_COLUMN}' contains null values.")
        unique_targets = set(tx_df[TARGET_COLUMN].unique().tolist())
        if not unique_targets.issubset({0, 1}):
            raise ValueError(f"Unexpected values in '{TARGET_COLUMN}': {sorted(unique_targets)}")

    logger.info("Loading IEEE-CIS %s identity records from %s", split, id_path)
    id_df = pd.read_csv(id_path, low_memory=False)

    if "TransactionID" not in id_df.columns:
        raise ValueError(f"{id_path.name} is missing join key 'TransactionID'.")

    rename_map = normalize_identity_columns(list(id_df.columns))
    if rename_map:
        id_df = id_df.rename(columns=rename_map)

    if id_df["TransactionID"].duplicated().any():
        logger.warning("Deduplicating duplicate TransactionID rows in %s", id_path.name)
        id_df = id_df.drop_duplicates(subset=["TransactionID"], keep="first")

    id_df["has_identity_record"] = 1
    merged = tx_df.merge(id_df, on="TransactionID", how="left", validate="one_to_one")
    merged["has_identity_record"] = merged["has_identity_record"].fillna(0).astype("int8")

    # Enforce strict chronological ordering to prevent temporal target leakage
    merged = merged.sort_values(
        by=["TransactionDT", "TransactionID"], ascending=[True, True]
    ).reset_index(drop=True)

    logger.info(
        "Merged IEEE-CIS %s dataset: %d rows, %d columns, identity match rate %.2f%%",
        split,
        len(merged),
        merged.shape[1],
        float(merged["has_identity_record"].mean() * 100.0),
    )
    return merged


def chronological_train_val_split(
    df: "pd.DataFrame",
    train_fraction: float = 0.80,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Split transactions strictly chronologically by TransactionDT.

    Random shuffling leaks future card/device behavior into historical training
    rows. A time-based holdout mirrors production deployment where models train
    on past transactions and score future transactions.
    """
    if not 0.5 <= train_fraction < 0.99:
        raise ValueError(f"train_fraction must be in [0.5, 0.99), got {train_fraction}")
    if "TransactionDT" not in df.columns:
        raise ValueError("DataFrame must include 'TransactionDT' for chronological splitting.")

    ordered = df.sort_values(
        by=["TransactionDT", "TransactionID"], ascending=[True, True]
    ).reset_index(drop=True)

    split_idx = int(len(ordered) * train_fraction)
    if split_idx <= 0 or split_idx >= len(ordered):
        raise ValueError(f"Dataset too small ({len(ordered)} rows) for train_fraction={train_fraction}.")

    train_df = ordered.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = ordered.iloc[split_idx:].copy().reset_index(drop=True)

    max_train_dt = float(train_df["TransactionDT"].max())
    min_val_dt = float(val_df["TransactionDT"].min())
    if max_train_dt > min_val_dt:
        raise RuntimeError(
            f"Temporal leakage detected: max train TransactionDT ({max_train_dt}) > min val ({min_val_dt})."
        )

    return train_df, val_df
