"""Phase 1 unit & integration tests for dataset validation, causal features, evaluation, and XGBoost."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ML_SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(ML_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_ROOT))

from src.config import Phase1Config
from src.data.loader import (
    MissingDatasetError,
    normalize_identity_columns,
    validate_required_files,
)
from src.features.engineering import (
    build_entity_keys_row,
    compute_causal_behavioral_records,
)
from src.models.evaluate import (
    compute_confusion_counts,
    optimize_thresholds_from_pr,
)

HAS_PANDAS_XGB = all(
    importlib.util.find_spec(pkg) is not None
    for pkg in ("pandas", "numpy", "sklearn", "xgboost", "joblib")
)


class TestDatasetGuardsAndNormalization(unittest.TestCase):
    def test_missing_dataset_raises_actionable_error_without_fabricating_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MissingDatasetError) as ctx:
                validate_required_files(Path(tmp), split="train")
            msg = str(ctx.exception)
            self.assertIn("train_transaction.csv", msg)
            self.assertIn("train_identity.csv", msg)
            self.assertIn("kaggle.com/competitions/ieee-fraud-detection", msg)

    def test_invalid_split_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                validate_required_files(Path(tmp), split="validation")

    def test_normalize_test_identity_hyphenated_columns(self) -> None:
        cols = ["TransactionID", "id-01", "id-12", "id-38", "DeviceType", "DeviceInfo"]
        rename_map = normalize_identity_columns(cols)
        self.assertEqual(rename_map, {"id-01": "id_01", "id-12": "id_12", "id-38": "id_38"})


class TestCausalFeatureEngineering(unittest.TestCase):
    def test_strictly_causal_velocity_amount_deviation_and_entity_shifts(self) -> None:
        # Deliberately pass rows out of chronological order to verify sorting
        records = [
            {
                "TransactionID": 3,
                "TransactionDT": 4000,  # >1h after tx1 (100), <1h after tx2 (1000)
                "TransactionAmt": 500.0,
                "card1": 1001,
                "addr1": 325,  # Location shifted from 200 -> 325
                "addr2": 87,
                "P_emaildomain": "gmail.com",
                "DeviceInfo": "SM-G991B",  # Device shifted from iOS -> Android
                "DeviceType": "mobile",
                "has_identity_record": 1,
            },
            {
                "TransactionID": 1,
                "TransactionDT": 100,
                "TransactionAmt": 100.0,
                "card1": 1001,
                "addr1": 200,
                "addr2": 87,
                "P_emaildomain": "gmail.com",
                "DeviceInfo": "iOS Device",
                "DeviceType": "mobile",
                "has_identity_record": 1,
            },
            {
                "TransactionID": 2,
                "TransactionDT": 1000,  # 900s after tx1
                "TransactionAmt": 100.0,
                "card1": 1001,
                "addr1": 200,
                "addr2": 87,
                "P_emaildomain": "gmail.com",
                "DeviceInfo": "iOS Device",
                "DeviceType": "mobile",
                "has_identity_record": 1,
            },
        ]

        feats = compute_causal_behavioral_records(records)
        # Results align to original input indices
        f_tx3, f_tx1, f_tx2 = feats[0], feats[1], feats[2]

        # Tx1 is first observation for this account
        self.assertEqual(f_tx1["account_tx_count_prior"], 0)
        self.assertEqual(f_tx1["tx_count_1h"], 0)
        self.assertEqual(f_tx1["time_since_prev_tx_sec"], -1.0)
        self.assertEqual(f_tx1["amount_vs_user_avg"], 1.0)
        self.assertEqual(f_tx1["device_change_flag"], 0)
        self.assertEqual(f_tx1["location_change_flag"], 0)

        # Tx2 occurs 900s after Tx1 on identical device & location
        self.assertEqual(f_tx2["account_tx_count_prior"], 1)
        self.assertEqual(f_tx2["tx_count_1h"], 1)
        self.assertEqual(f_tx2["time_since_prev_tx_sec"], 900.0)
        self.assertEqual(f_tx2["account_amt_mean_prior"], 100.0)
        self.assertEqual(f_tx2["amount_vs_user_avg"], 1.0)
        self.assertEqual(f_tx2["device_change_flag"], 0)
        self.assertEqual(f_tx2["location_change_flag"], 0)

        # Tx3 occurs at t=4000: tx1 (t=100) aged out of 1h window (3900s > 3600s), tx2 (t=1000) remains
        # Note: addr1 changed so account_key changed; verify device/location behavior for same account too
        account_same_records = [
            {
                "TransactionID": 10,
                "TransactionDT": 100,
                "TransactionAmt": 50.0,
                "card1": 5555,
                "addr1": 200,
                "addr2": 87,
                "P_emaildomain": "yahoo.com",
                "DeviceInfo": "MacOS",
                "DeviceType": "desktop",
            },
            {
                "TransactionID": 11,
                "TransactionDT": 1900,
                "TransactionAmt": 250.0,
                "card1": 5555,
                "addr1": 200,
                "addr2": 88,  # addr2 shifted -> location_change_flag = 1 while same account_key
                "P_emaildomain": "yahoo.com",
                "DeviceInfo": "Windows",  # Device shifted -> device_change_flag = 1
                "DeviceType": "desktop",
            },
        ]
        same_feats = compute_causal_behavioral_records(account_same_records)
        self.assertEqual(same_feats[1]["tx_count_1h"], 1)
        self.assertEqual(same_feats[1]["account_amt_mean_prior"], 50.0)
        self.assertEqual(same_feats[1]["amount_vs_user_avg"], 5.0)
        self.assertEqual(same_feats[1]["time_since_prev_tx_sec"], 1800.0)
        self.assertEqual(same_feats[1]["device_change_flag"], 1)
        self.assertEqual(same_feats[1]["location_change_flag"], 1)
        # Ensure f_tx3 still computed valid log1p and hour
        self.assertGreater(float(f_tx3["amt_log1p"]), 6.0)

    def test_entity_keys_handle_missing_values(self) -> None:
        acc, dev, loc = build_entity_keys_row(None, float("nan"), "", None, None, None)
        self.assertEqual(acc, "c1:UNKNOWN|a1:UNKNOWN|em:UNKNOWN")
        self.assertEqual(dev, "UNKNOWN|UNKNOWN")
        self.assertEqual(loc, "UNKNOWN|UNKNOWN")


class TestEvaluationMetrics(unittest.TestCase):
    def test_confusion_counts_exact(self) -> None:
        y_true = [1, 1, 0, 0, 0, 1]
        y_prob = [0.92, 0.40, 0.85, 0.10, 0.20, 0.78]
        cm = compute_confusion_counts(y_true, y_prob, threshold=0.75)
        self.assertEqual(cm, {"tp": 2, "fp": 1, "fn": 1, "tn": 2})

    def test_threshold_optimization_respects_order(self) -> None:
        precisions = [0.10, 0.45, 0.72, 0.90, 1.0]
        recalls = [1.0, 0.90, 0.78, 0.45, 0.0]
        thresholds = [0.15, 0.40, 0.65, 0.85]
        res = optimize_thresholds_from_pr(precisions, recalls, thresholds, min_precision_for_block=0.50)
        self.assertEqual(res.block_threshold, 0.65)
        self.assertLess(res.review_threshold, res.block_threshold)
        self.assertGreater(res.best_f1, 0.70)


@unittest.skipUnless(HAS_PANDAS_XGB, "Requires pandas, numpy, scikit-learn, xgboost, joblib")
class TestPhase1EndToEndWithMlDeps(unittest.TestCase):
    def test_chronological_split_and_xgboost_training_roundtrip(self) -> None:
        import joblib
        import numpy as np
        import pandas as pd

        from src.data.loader import chronological_train_val_split
        from src.models.train_xgboost import train_xgboost_pipeline

        rng = np.random.default_rng(42)
        n = 400
        tx_ids = np.arange(1, n + 1)
        tx_dt = np.arange(1, n + 1) * 300
        # Create minimal unit-test fixture CSVs inside a temporary directory
        is_fraud = np.zeros(n, dtype=int)
        is_fraud[rng.choice(n, size=32, replace=False)] = 1
        amt = np.where(is_fraud == 1, rng.uniform(400, 1500, n), rng.uniform(10, 150, n))

        tx_df = pd.DataFrame(
            {
                "TransactionID": tx_ids,
                "TransactionDT": tx_dt,
                "TransactionAmt": np.round(amt, 2),
                "isFraud": is_fraud,
                "ProductCD": np.where(is_fraud == 1, "C", "W"),
                "card1": rng.integers(1000, 1040, n),
                "card4": rng.choice(["visa", "mastercard"], n),
                "card6": rng.choice(["debit", "credit"], n),
                "addr1": rng.integers(200, 215, n),
                "addr2": 87,
                "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "anonymous.com"], n),
            }
        )
        id_df = pd.DataFrame(
            {
                "TransactionID": tx_ids[::2],
                "DeviceType": rng.choice(["mobile", "desktop"], n // 2),
                "DeviceInfo": rng.choice(["iOS Device", "Windows", "SM-G991B"], n // 2),
                "id_01": rng.normal(0, 5, n // 2),
            }
        )

        train_part, val_part = chronological_train_val_split(tx_df, train_fraction=0.80)
        self.assertLessEqual(train_part["TransactionDT"].max(), val_part["TransactionDT"].min())

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            out_dir = tmp_path / "out"
            raw_dir.mkdir(parents=True)
            tx_df.to_csv(raw_dir / "train_transaction.csv", index=False)
            id_df.to_csv(raw_dir / "train_identity.csv", index=False)

            cfg = Phase1Config(
                data_dir=raw_dir,
                eda_output_dir=tmp_path / "eda",
                model_output_dir=out_dir,
                train_fraction=0.80,
                random_seed=42,
                xgb_n_estimators=25,
                xgb_max_depth=3,
                xgb_learning_rate=0.1,
                xgb_subsample=0.8,
                xgb_colsample_bytree=0.8,
                xgb_early_stopping_rounds=10,
                mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                mlflow_experiment_name="test-exp",
            )
            report = train_xgboost_pipeline(cfg=cfg, log_mlflow=False, model_version="test-v1")
            self.assertIn("pr_auc", report["metrics"])
            self.assertTrue((out_dir / "model_bundle.joblib").is_file())
            self.assertTrue((out_dir / "metrics.json").is_file())
            self.assertTrue((out_dir / "pr_curve.json").is_file())

            loaded = joblib.load(out_dir / "model_bundle.joblib")
            self.assertEqual(loaded["model_version"], "test-v1")
            metrics_disk = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics_disk["model_version"], "test-v1")


if __name__ == "__main__":
    unittest.main()
