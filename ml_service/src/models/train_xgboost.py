"""Genuine XGBoost training, evaluation, and artifact persistence pipeline.

Trains an imbalanced binary XGBoost classifier on merged IEEE-CIS transactions
using:
  - Leakage-free causal behavioral feature engineering
  - Chronological train/validation split (by TransactionDT)
  - Exact class-imbalance weighting (scale_pos_weight = n_neg / n_pos)
  - Optional SMOTE oversampling experiment (--use-smote, fit on train fold only)
  - Early stopping monitored on validation PR-AUC (aucpr)
  - Serialized model + preprocessor bundle via joblib + measured metrics JSON
  - Optional MLflow logging (--log-mlflow)

Run:
    cd ml_service
    python -m src.models.train_xgboost [--nrows 100000] [--use-smote] [--log-mlflow]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import Phase1Config, load_config
from src.data.loader import TARGET_COLUMN, chronological_train_val_split, load_ieee_cis_split
from src.features.engineering import FraudFeaturePreprocessor, engineer_transaction_features
from src.models.evaluate import evaluate_binary_predictions

logger = logging.getLogger(__name__)


def train_xgboost_pipeline(
    cfg: Phase1Config,
    nrows: int | None = None,
    use_smote: bool = False,
    log_mlflow: bool = False,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Execute end-to-end Phase 1 XGBoost training and save versioned artifacts."""
    import joblib
    import numpy as np
    import xgboost as xgb

    started = time.perf_counter()
    version_tag = model_version or datetime.now(timezone.utc).strftime("xgb-%Y%m%d-%H%M%S")

    # 1. Load real merged IEEE-CIS dataset
    raw_df = load_ieee_cis_split(data_dir=cfg.data_dir, split="train", nrows=nrows)

    # 2. Compute strictly causal behavioral features across chronological stream
    logger.info("Engineering causal transaction velocity, amount deviation, and device/location shift features...")
    enriched_df = engineer_transaction_features(raw_df)

    # 3. Chronological 80/20 train-validation split (zero future leakage)
    train_df, val_df = chronological_train_val_split(enriched_df, train_fraction=cfg.train_fraction)
    y_train = train_df[TARGET_COLUMN].astype(np.int32)
    y_val = val_df[TARGET_COLUMN].astype(np.int32)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0 or int((y_val == 1).sum()) == 0:
        raise ValueError(
            "Both training and validation chronological splits must contain positive fraud examples."
        )

    # 4. Fit leakage-safe preprocessor strictly on training split (OOF target encoding on X_train)
    preprocessor = FraudFeaturePreprocessor(random_state=cfg.random_seed)
    X_train = preprocessor.fit_transform(train_df, y_train)
    X_val = preprocessor.transform(val_df)

    # 5. Handle class imbalance: scale_pos_weight or optional SMOTE experiment
    scale_pos_weight = float(n_neg / max(1, n_pos))
    imbalance_strategy = f"scale_pos_weight={scale_pos_weight:.2f}"
    X_fit, y_fit = X_train, y_train

    if use_smote:
        from imblearn.over_sampling import SMOTE

        logger.info("Applying SMOTE oversampling strictly to training split (sampling_strategy=0.10)...")
        smote = SMOTE(sampling_strategy=0.10, random_state=cfg.random_seed)
        X_fit, y_fit = smote.fit_resample(X_train, y_train)
        scale_pos_weight = 1.0
        imbalance_strategy = "SMOTE(sampling_strategy=0.10)"

    # 6. Train XGBoost binary classifier optimizing PR-AUC (aucpr)
    params: dict[str, Any] = {
        "n_estimators": cfg.xgb_n_estimators,
        "max_depth": cfg.xgb_max_depth,
        "learning_rate": cfg.xgb_learning_rate,
        "subsample": cfg.xgb_subsample,
        "colsample_bytree": cfg.xgb_colsample_bytree,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": cfg.random_seed,
        "n_jobs": -1,
        "early_stopping_rounds": cfg.xgb_early_stopping_rounds,
    }

    logger.info(
        "Training XGBoost (%s) on %d train rows / %d val rows (%d features)...",
        imbalance_strategy,
        len(X_fit),
        len(X_val),
        X_fit.shape[1],
    )
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 7. Evaluate validation set using PR-AUC & threshold optimization
    val_probs = model.predict_proba(X_val)[:, 1]
    out_dir = cfg.model_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_metrics, pr_curve = evaluate_binary_predictions(
        y_true=y_val.values,
        y_prob=val_probs,
        chart_output_path=out_dir / "pr_roc_curves.png",
    )

    # Extract top feature importances from trained booster
    importances = model.feature_importances_
    top_features = sorted(
        [
            {"feature": fname, "importance": round(float(imp), 6)}
            for fname, imp in zip(preprocessor.output_feature_names_, importances)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )[:25]

    duration_sec = round(time.perf_counter() - started, 2)
    report: dict[str, Any] = {
        "model_version": version_tag,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "IEEE-CIS Fraud Detection",
        "rows_total": int(len(raw_df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "train_fraud_count": n_pos,
        "train_legit_count": n_neg,
        "val_fraud_count": int((y_val == 1).sum()),
        "feature_count": int(len(preprocessor.output_feature_names_)),
        "imbalance_strategy": imbalance_strategy,
        "best_iteration": int(getattr(model, "best_iteration", cfg.xgb_n_estimators - 1)),
        "training_duration_sec": duration_sec,
        "hyperparameters": params,
        "metrics": eval_metrics,
        "top_feature_importances": top_features,
    }

    # 8. Persist model bundle, preprocessor, feature schema, metrics, and PR curve
    bundle = {
        "model_version": version_tag,
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": preprocessor.output_feature_names_,
        "block_threshold": eval_metrics["optimized_block_threshold"],
        "review_threshold": eval_metrics["optimized_review_threshold"],
        "metrics": eval_metrics,
        "trained_at_utc": report["trained_at_utc"],
    }
    bundle_path = out_dir / "model_bundle.joblib"
    joblib.dump(bundle, bundle_path)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    pr_curve_path = out_dir / "pr_curve.json"
    pr_curve_path.write_text(json.dumps(pr_curve, indent=2), encoding="utf-8")

    schema_path = out_dir / "feature_schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "model_version": version_tag,
                "feature_names": preprocessor.output_feature_names_,
                "numeric_columns": preprocessor.numeric_columns_,
                "categorical_columns": preprocessor.categorical_columns_,
                "risk_encoded_categories": list(preprocessor.risk_rate_maps_.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Saved XGBoost model bundle to %s | Val PR-AUC=%.5f | ROC-AUC=%.5f | F1=%.5f (thr=%.4f)",
        bundle_path,
        eval_metrics["pr_auc"],
        eval_metrics["roc_auc"],
        eval_metrics["f1_score"],
        eval_metrics["optimized_block_threshold"],
    )

    # 9. Optional MLflow tracking
    if log_mlflow:
        _log_run_to_mlflow(cfg=cfg, model=model, report=report, out_dir=out_dir)

    return report


def _log_run_to_mlflow(
    cfg: Phase1Config,
    model: Any,
    report: dict[str, Any],
    out_dir: Path,
) -> None:
    import mlflow
    import mlflow.xgboost

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    with mlflow.start_run(run_name=report["model_version"]):
        mlflow.log_params(
            {
                k: v
                for k, v in report["hyperparameters"].items()
                if isinstance(v, (int, float, str, bool))
            }
        )
        mlflow.log_param("imbalance_strategy", report["imbalance_strategy"])
        mlflow.log_param("rows_train", report["rows_train"])
        mlflow.log_param("rows_val", report["rows_val"])
        mlflow.log_param("feature_count", report["feature_count"])

        m = report["metrics"]
        mlflow.log_metrics(
            {
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "precision_at_90_recall": m["precision_at_90_recall"],
                "optimized_block_threshold": m["optimized_block_threshold"],
                "training_duration_sec": report["training_duration_sec"],
            }
        )
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")
        for artifact in ("metrics.json", "pr_curve.json", "feature_schema.json", "pr_roc_curves.png"):
            p = out_dir / artifact
            if p.is_file():
                mlflow.log_artifact(str(p))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train SentinelRisk XGBoost fraud detection model.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Raw IEEE-CIS CSV directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for trained artifacts.")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for local debugging.")
    parser.add_argument("--use-smote", action="store_true", help="Use SMOTE on training fold instead of scale_pos_weight.")
    parser.add_argument("--log-mlflow", action="store_true", help="Log parameters, metrics, and model artifacts to MLflow.")
    parser.add_argument("--version", type=str, default=None, help="Optional explicit model version tag.")
    args = parser.parse_args()

    cfg = load_config()
    if args.data_dir or args.output_dir:
        cfg = Phase1Config(
            **{
                **cfg.__dict__,
                "data_dir": args.data_dir or cfg.data_dir,
                "model_output_dir": args.output_dir or cfg.model_output_dir,
            }
        )

    report = train_xgboost_pipeline(
        cfg=cfg,
        nrows=args.nrows,
        use_smote=args.use_smote,
        log_mlflow=args.log_mlflow,
        model_version=args.version,
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
