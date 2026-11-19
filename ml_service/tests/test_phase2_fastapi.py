"""Phase 2 automated tests for FastAPI inference service, model loading, and validation."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ML_SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(ML_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_ROOT))

from src.api.model_service import (
    FeatureSchemaError,
    ModelInferenceService,
    ModelNotReadyError,
    TransactionFeaturesNotFoundError,
    classify_decision,
    normalize_input_to_ieee_row,
)

HAS_FASTAPI_STACK = all(
    importlib.util.find_spec(pkg) is not None
    for pkg in ("fastapi", "httpx", "pandas", "numpy", "joblib")
)


class TestModelServiceCoreStdlib(unittest.TestCase):
    """Pure-Python unit tests executable without external ML/FastAPI wheels."""

    def test_missing_artifacts_marks_service_not_ready_and_raises_on_predict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            svc = ModelInferenceService(artifact_dir=Path(tmp))
            loaded = svc.load_artifacts()
            self.assertFalse(loaded)
            self.assertFalse(svc.is_ready)
            self.assertTrue(any("model_bundle.joblib" in p for p in svc.missing_artifacts))

            health = svc.get_health()
            self.assertEqual(health["status"], "degraded")
            self.assertFalse(health["model_ready"])

            info = svc.get_model_info()
            self.assertFalse(info["ready"])
            self.assertIsNone(info["model_version"])

            with self.assertRaises(ModelNotReadyError) as ctx:
                svc.predict_one(
                    {"transaction_id": "TX-1", "transaction_dt": 100.0, "transaction_amt": 50.0}
                )
            self.assertTrue(any("model_bundle.joblib" in p for p in ctx.exception.missing_artifacts))

    def test_decision_thresholds(self) -> None:
        self.assertEqual(classify_decision(0.12, review_threshold=0.50, block_threshold=0.80), "Approved")
        self.assertEqual(classify_decision(0.65, review_threshold=0.50, block_threshold=0.80), "Review")
        self.assertEqual(classify_decision(0.84, review_threshold=0.50, block_threshold=0.80), "Blocked")

    def test_payload_normalization_and_invalid_schema_errors(self) -> None:
        valid = normalize_input_to_ieee_row(
            {
                "transaction_id": "TX-200",
                "transaction_dt": 86400,
                "transaction_amt": 125.75,
                "card1": 13926,
                "device_type": "mobile",
            }
        )
        self.assertEqual(valid["TransactionID"], "TX-200")
        self.assertEqual(valid["TransactionAmt"], 125.75)
        self.assertEqual(valid["has_identity_record"], 1)

        with self.assertRaises(FeatureSchemaError):
            normalize_input_to_ieee_row({"transaction_id": "", "transaction_dt": 10, "transaction_amt": 5})

        with self.assertRaises(FeatureSchemaError):
            normalize_input_to_ieee_row({"transaction_id": "TX-1", "transaction_dt": 10, "transaction_amt": -5})

        with self.assertRaises(FeatureSchemaError):
            normalize_input_to_ieee_row(
                {
                    "transaction_id": "TX-1",
                    "transaction_dt": 10,
                    "transaction_amt": 50,
                    "extra_features": {"V1": float("nan")},
                }
            )

    def test_features_lookup_raises_not_found_for_unknown_tx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            svc = ModelInferenceService(artifact_dir=Path(tmp))
            with self.assertRaises(TransactionFeaturesNotFoundError):
                svc.get_transaction_features("UNKNOWN-TX")


class _MockPreprocessor:
    """Test-only deterministic preprocessor stub used strictly in automated FastAPI tests."""

    def __init__(self, feature_names: list[str], corrupt_columns: bool = False) -> None:
        self.output_feature_names_ = list(feature_names)
        self._corrupt_columns = corrupt_columns

    def transform(self, df: Any) -> Any:
        import pandas as pd

        out = pd.DataFrame(index=df.index)
        cols = list(reversed(self.output_feature_names_)) if self._corrupt_columns else self.output_feature_names_
        for col in cols:
            if col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")
            else:
                out[col] = 0.0
        return out


class _MockBinaryModel:
    """Test-only deterministic predict_proba stub used strictly in automated FastAPI tests."""

    def __init__(self) -> None:
        self.feature_importances_ = [0.6, 0.25, 0.15]

    def predict_proba(self, X: Any) -> Any:
        import numpy as np

        probs = []
        for _, row in X.iterrows():
            amt = float(row.get("TransactionAmt", 0.0))
            p = 0.91 if amt >= 1000.0 else 0.62 if amt >= 300.0 else 0.08
            probs.append([1.0 - p, p])
        return np.asarray(probs, dtype=np.float64)


@unittest.skipUnless(HAS_FASTAPI_STACK, "Requires fastapi, httpx, pandas, numpy, joblib")
class TestFastApiEndpoints(unittest.TestCase):
    """Full HTTP integration tests against all 7 FastAPI endpoints using TestClient."""

    def _write_test_bundle(self, artifact_dir: Path, version: str = "xgb-test-v1", corrupt: bool = False) -> None:
        import joblib

        feature_names = ["TransactionAmt", "amount_vs_user_avg", "tx_count_1h"]
        bundle = {
            "model_version": version,
            "model": _MockBinaryModel(),
            "preprocessor": _MockPreprocessor(feature_names, corrupt_columns=corrupt),
            "feature_names": feature_names,
            "review_threshold": 0.50,
            "block_threshold": 0.80,
            "trained_at_utc": "2026-11-19T00:00:00+00:00",
            "metrics": {"pr_auc": 0.8123, "roc_auc": 0.9456, "f1_score": 0.7421},
        }
        joblib.dump(bundle, artifact_dir / "model_bundle.joblib")

    def test_model_unavailable_returns_503_and_reload_activates_model(self) -> None:
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        with tempfile.TemporaryDirectory() as tmp:
            art_dir = Path(tmp)
            app = create_app(artifact_dir=art_dir)
            with TestClient(app) as client:
                # 1. Health & readiness report degraded / not ready
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json()["status"], "degraded")
                self.assertFalse(health.json()["model_ready"])

                info = client.get("/model/info")
                self.assertEqual(info.status_code, 200)
                self.assertFalse(info.json()["ready"])

                # 2. Predict returns HTTP 503 model_not_ready
                pred_unready = client.post(
                    "/predict",
                    json={"transaction_id": "TX-503", "transaction_dt": 1000, "transaction_amt": 250.0},
                )
                self.assertEqual(pred_unready.status_code, 503)
                self.assertEqual(pred_unready.json()["error"], "model_not_ready")

                # 3. Write artifact and trigger POST /model/reload
                self._write_test_bundle(art_dir, version="xgb-reloaded-v2")
                reload_res = client.post("/model/reload")
                self.assertEqual(reload_res.status_code, 200)
                self.assertTrue(reload_res.json()["ready"])
                self.assertEqual(reload_res.json()["model_version"], "xgb-reloaded-v2")

                # 4. Predict now succeeds with real model_version
                pred_ok = client.post(
                    "/predict",
                    json={
                        "transaction_id": "TX-200",
                        "transaction_dt": 1200,
                        "transaction_amt": 1450.0,
                        "card1": 13926,
                    },
                )
                self.assertEqual(pred_ok.status_code, 200)
                body = pred_ok.json()
                self.assertEqual(body["transaction_id"], "TX-200")
                self.assertEqual(body["decision"], "Blocked")
                self.assertEqual(body["model_version"], "xgb-reloaded-v2")
                self.assertAlmostEqual(body["xgboost_probability"], 0.91, places=3)

    def test_valid_batch_features_metrics_and_validation_errors(self) -> None:
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        with tempfile.TemporaryDirectory() as tmp:
            art_dir = Path(tmp)
            self._write_test_bundle(art_dir, version="xgb-test-v1")
            app = create_app(artifact_dir=art_dir)

            with TestClient(app) as client:
                # Valid single prediction
                r1 = client.post(
                    "/predict",
                    json={"transaction_id": "TX-A1", "transaction_dt": 500, "transaction_amt": 45.0, "card1": 1001},
                )
                self.assertEqual(r1.status_code, 200)
                self.assertEqual(r1.json()["decision"], "Approved")

                # Invalid payload (negative amount & missing required fields)
                bad_amt = client.post(
                    "/predict",
                    json={"transaction_id": "TX-BAD", "transaction_dt": 500, "transaction_amt": -10.0},
                )
                self.assertEqual(bad_amt.status_code, 422)

                missing_fields = client.post("/predict", json={"transaction_id": "TX-MISS"})
                self.assertEqual(missing_fields.status_code, 422)

                # Batch inference
                batch = client.post(
                    "/predict/batch",
                    json={
                        "transactions": [
                            {"transaction_id": "TX-B1", "transaction_dt": 600, "transaction_amt": 420.0, "card1": 1001},
                            {"transaction_id": "TX-B2", "transaction_dt": 700, "transaction_amt": 2100.0, "card1": 1001},
                        ]
                    },
                )
                self.assertEqual(batch.status_code, 200)
                b_json = batch.json()
                self.assertEqual(b_json["count"], 2)
                self.assertEqual(b_json["predictions"][0]["decision"], "Review")
                self.assertEqual(b_json["predictions"][1]["decision"], "Blocked")

                # Feature lookup (200 for scored tx, 404 for unknown)
                feat_ok = client.get("/features/TX-B2")
                self.assertEqual(feat_ok.status_code, 200)
                self.assertEqual(feat_ok.json()["transaction_id"], "TX-B2")
                self.assertIn("tx_count_1h", feat_ok.json()["engineered_features"])

                feat_404 = client.get("/features/TX-DOES-NOT-EXIST")
                self.assertEqual(feat_404.status_code, 404)

                # Runtime metrics reflect actual predictions
                metrics = client.get("/metrics")
                self.assertEqual(metrics.status_code, 200)
                m = metrics.json()
                self.assertEqual(m["total_predictions"], 3)
                self.assertEqual(m["batch_requests"], 1)
                self.assertEqual(m["decision_counts"], {"Approved": 1, "Review": 1, "Blocked": 1})

    def test_incorrect_feature_schema_order_returns_422(self) -> None:
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        with tempfile.TemporaryDirectory() as tmp:
            art_dir = Path(tmp)
            self._write_test_bundle(art_dir, version="xgb-corrupt", corrupt=True)
            app = create_app(artifact_dir=art_dir)

            with TestClient(app) as client:
                res = client.post(
                    "/predict",
                    json={"transaction_id": "TX-SCHEMA", "transaction_dt": 900, "transaction_amt": 99.0},
                )
                self.assertEqual(res.status_code, 422)
                self.assertEqual(res.json()["error"], "invalid_feature_schema")


if __name__ == "__main__":
    unittest.main()
