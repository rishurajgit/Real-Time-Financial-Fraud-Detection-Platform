"""Central configuration for the SentinelRisk Python ML service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional when environment variables are injected directly
    pass


ML_SERVICE_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(env_var: str, default_relative: str) -> Path:
    raw = os.getenv(env_var)
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else (ML_SERVICE_ROOT / p).resolve()
    return (ML_SERVICE_ROOT / default_relative).resolve()


@dataclass(frozen=True)
class Phase1Config:
    data_dir: Path
    eda_output_dir: Path
    model_output_dir: Path
    train_fraction: float
    random_seed: int
    xgb_n_estimators: int
    xgb_max_depth: int
    xgb_learning_rate: float
    xgb_subsample: float
    xgb_colsample_bytree: float
    xgb_early_stopping_rounds: int
    mlflow_tracking_uri: str
    mlflow_experiment_name: str


def load_config() -> Phase1Config:
    """Build immutable Phase 1 configuration from environment variables."""
    return Phase1Config(
        data_dir=_resolve_path("SENTINEL_DATA_DIR", "data/raw"),
        eda_output_dir=_resolve_path("SENTINEL_EDA_OUTPUT_DIR", "artifacts/eda"),
        model_output_dir=_resolve_path("SENTINEL_MODEL_OUTPUT_DIR", "artifacts/models/xgboost"),
        train_fraction=float(os.getenv("SENTINEL_TRAIN_FRACTION", "0.80")),
        random_seed=int(os.getenv("SENTINEL_RANDOM_SEED", "42")),
        xgb_n_estimators=int(os.getenv("SENTINEL_XGB_N_ESTIMATORS", "600")),
        xgb_max_depth=int(os.getenv("SENTINEL_XGB_MAX_DEPTH", "6")),
        xgb_learning_rate=float(os.getenv("SENTINEL_XGB_LEARNING_RATE", "0.05")),
        xgb_subsample=float(os.getenv("SENTINEL_XGB_SUBSAMPLE", "0.80")),
        xgb_colsample_bytree=float(os.getenv("SENTINEL_XGB_COLSAMPLE_BYTREE", "0.80")),
        xgb_early_stopping_rounds=int(os.getenv("SENTINEL_XGB_EARLY_STOPPING", "40")),
        mlflow_tracking_uri=os.getenv(
            "MLFLOW_TRACKING_URI", f"sqlite:///{(ML_SERVICE_ROOT / 'mlflow.db').as_posix()}"
        ),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "sentinelrisk-ieee-cis-xgboost"),
    )
