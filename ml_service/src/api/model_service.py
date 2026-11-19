"""Real XGBoost model loading, feature alignment, causal state tracking, and inference service.

Reuses the exact Phase 1 preprocessing artifact (FraudFeaturePreprocessor) and
causal feature definitions (src/features/engineering.py) to prevent training-serving
skew. Never fabricates predictions when model artifacts are missing.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import load_config
from src.features.engineering import build_entity_keys_row

logger = logging.getLogger(__name__)

REQUIRED_BUNDLE_FILENAME = "model_bundle.joblib"
METRICS_FILENAME = "metrics.json"
PR_CURVE_FILENAME = "pr_curve.json"
SCHEMA_FILENAME = "feature_schema.json"


class ModelNotReadyError(RuntimeError):
    """Raised when prediction is attempted without valid trained model artifacts."""

    def __init__(self, message: str, missing_artifacts: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing_artifacts = missing_artifacts or []


class FeatureSchemaError(ValueError):
    """Raised when incoming transaction features violate expected schema or contain invalid values."""


class TransactionFeaturesNotFoundError(KeyError):
    """Raised when /features/{transaction_id} is requested for an unknown transaction."""


@dataclass
class AccountCausalState:
    window_1h: deque[float] = field(default_factory=deque)
    window_24h: deque[float] = field(default_factory=deque)
    window_7d: deque[float] = field(default_factory=deque)
    amt_count: int = 0
    amt_mean: float = 0.0
    amt_m2: float = 0.0
    last_dt: float | None = None
    last_device: str | None = None
    last_location: str | None = None


def classify_decision(
    probability: float,
    review_threshold: float,
    block_threshold: float,
) -> str:
    """Map calibrated fraud probability to Approved / Review / Blocked."""
    if probability >= block_threshold:
        return "Blocked"
    if probability >= review_threshold:
        return "Review"
    return "Approved"


def normalize_input_to_ieee_row(payload: dict[str, Any]) -> dict[str, Any]:
    """Map API snake_case payload into exact IEEE-CIS column names expected by Phase 1 preprocessor."""
    tx_id = str(payload.get("transaction_id", "")).strip()
    if not tx_id:
        raise FeatureSchemaError("Field 'transaction_id' is required and must be non-empty.")

    try:
        tx_dt = float(payload["transaction_dt"])
        tx_amt = float(payload["transaction_amt"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureSchemaError(
            "Numeric fields 'transaction_dt' and 'transaction_amt' are required."
        ) from exc

    if not math.isfinite(tx_dt) or tx_dt < 0:
        raise FeatureSchemaError("Field 'transaction_dt' must be a finite non-negative number.")
    if not math.isfinite(tx_amt) or tx_amt < 0:
        raise FeatureSchemaError("Field 'transaction_amt' must be a finite non-negative number.")

    has_identity = payload.get("has_identity_record")
    if has_identity is None:
        has_identity = 1 if (payload.get("device_type") or payload.get("device_info")) else 0

    row: dict[str, Any] = {
        "TransactionID": tx_id,
        "TransactionDT": tx_dt,
        "TransactionAmt": tx_amt,
        "ProductCD": payload.get("product_cd"),
        "card1": payload.get("card1"),
        "card2": payload.get("card2"),
        "card3": payload.get("card3"),
        "card4": payload.get("card4"),
        "card5": payload.get("card5"),
        "card6": payload.get("card6"),
        "addr1": payload.get("addr1"),
        "addr2": payload.get("addr2"),
        "dist1": payload.get("dist1"),
        "dist2": payload.get("dist2"),
        "P_emaildomain": payload.get("p_emaildomain"),
        "R_emaildomain": payload.get("r_emaildomain"),
        "DeviceType": payload.get("device_type"),
        "DeviceInfo": payload.get("device_info"),
        "has_identity_record": int(has_identity),
    }

    extra = payload.get("extra_features")
    if extra is not None:
        if not isinstance(extra, dict):
            raise FeatureSchemaError("'extra_features' must be a JSON object.")
        for key, value in extra.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise FeatureSchemaError(f"Feature '{key}' contains non-finite float value.")
            row[str(key)] = value

    return row


class ModelInferenceService:
    """Thread-safe lifecycle manager and real inference engine for trained Phase 1 XGBoost artifacts."""

    def __init__(
        self,
        artifact_dir: Path | None = None,
        review_threshold_override: float | None = None,
        block_threshold_override: float | None = None,
        max_feature_cache: int = 2000,
    ) -> None:
        cfg = load_config()
        self.artifact_dir = Path(artifact_dir) if artifact_dir else cfg.model_output_dir
        self._review_override = review_threshold_override
        self._block_override = block_threshold_override
        self._max_feature_cache = max_feature_cache

        self._lock = threading.RLock()
        self._started_at = time.monotonic()

        # Loaded artifact state
        self._ready = False
        self._model: Any = None
        self._preprocessor: Any = None
        self._feature_names: list[str] = []
        self._model_version: str | None = None
        self._trained_at_utc: str | None = None
        self._review_threshold: float = 0.50
        self._block_threshold: float = 0.80
        self._validation_metrics: dict[str, Any] | None = None
        self._pr_curve: list[dict[str, float]] | None = None
        self._feature_importances: dict[str, float] = {}
        self._missing_artifacts: list[str] = []

        # Online causal feature state & lookup cache
        self._account_states: dict[str, AccountCausalState] = {}
        self._feature_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

        # Runtime telemetry
        self._total_predictions = 0
        self._batch_requests = 0
        self._decision_counts = {"Approved": 0, "Review": 0, "Blocked": 0}
        self._error_count = 0
        self._latencies_ms: deque[float] = deque(maxlen=1000)

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def missing_artifacts(self) -> list[str]:
        with self._lock:
            return list(self._missing_artifacts)

    def uptime_seconds(self) -> float:
        return round(time.monotonic() - self._started_at, 2)

    def load_artifacts(self) -> bool:
        """Load Phase 1 model_bundle.joblib and optional evaluation reports from disk."""
        with self._lock:
            bundle_path = self.artifact_dir / REQUIRED_BUNDLE_FILENAME
            metrics_path = self.artifact_dir / METRICS_FILENAME
            pr_curve_path = self.artifact_dir / PR_CURVE_FILENAME

            missing: list[str] = []
            if not bundle_path.is_file():
                missing.append(str(bundle_path))

            if missing:
                self._reset_unready(missing)
                logger.warning(
                    "XGBoost model artifacts unavailable. Missing: %s", ", ".join(missing)
                )
                return False

            try:
                import joblib

                bundle = joblib.load(bundle_path)
            except Exception as exc:
                self._reset_unready([f"{bundle_path} (load failed: {exc})"])
                logger.exception("Failed to deserialize model bundle from %s", bundle_path)
                return False

            if not isinstance(bundle, dict):
                self._reset_unready([f"{bundle_path} (expected dict bundle)"])
                return False

            required_keys = ("model", "preprocessor", "feature_names", "model_version")
            absent_keys = [k for k in required_keys if k not in bundle or bundle[k] is None]
            if absent_keys:
                self._reset_unready([f"{bundle_path} missing bundle keys: {', '.join(absent_keys)}"])
                return False

            model = bundle["model"]
            preprocessor = bundle["preprocessor"]
            if not hasattr(model, "predict_proba") or not hasattr(preprocessor, "transform"):
                self._reset_unready([f"{bundle_path} invalid model or preprocessor object"])
                return False

            feature_names = list(bundle["feature_names"])
            preproc_names = list(getattr(preprocessor, "output_feature_names_", feature_names))
            if feature_names != preproc_names:
                self._reset_unready(
                    [f"{bundle_path} feature_names mismatch between bundle and preprocessor"]
                )
                return False

            # Threshold resolution: env override > explicit override > Phase 1 calibrated bundle thresholds
            env_review = os.getenv("SENTINEL_REVIEW_THRESHOLD")
            env_block = os.getenv("SENTINEL_BLOCK_THRESHOLD")
            review_thr = (
                float(env_review)
                if env_review
                else self._review_override
                if self._review_override is not None
                else float(bundle.get("review_threshold", 0.50))
            )
            block_thr = (
                float(env_block)
                if env_block
                else self._block_override
                if self._block_override is not None
                else float(bundle.get("block_threshold", 0.80))
            )
            if not (0.0 < review_thr < block_thr < 1.0):
                review_thr = min(max(review_thr, 0.05), 0.90)
                block_thr = min(max(block_thr, review_thr + 0.05), 0.98)

            validation_metrics: dict[str, Any] | None = bundle.get("metrics")
            if metrics_path.is_file():
                try:
                    validation_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                except Exception:
                    logger.warning("Could not parse %s; falling back to bundle metrics.", metrics_path)

            pr_curve: list[dict[str, float]] | None = None
            if pr_curve_path.is_file():
                try:
                    pr_curve = json.loads(pr_curve_path.read_text(encoding="utf-8"))
                except Exception:
                    logger.warning("Could not parse %s.", pr_curve_path)

            importances: dict[str, float] = {}
            raw_imp = getattr(model, "feature_importances_", None)
            if raw_imp is not None and len(raw_imp) == len(feature_names):
                importances = {
                    fname: round(float(val), 6) for fname, val in zip(feature_names, raw_imp)
                }

            self._model = model
            self._preprocessor = preprocessor
            self._feature_names = feature_names
            self._model_version = str(bundle["model_version"])
            self._trained_at_utc = bundle.get("trained_at_utc")
            self._review_threshold = round(review_thr, 4)
            self._block_threshold = round(block_thr, 4)
            self._validation_metrics = validation_metrics
            self._pr_curve = pr_curve
            self._feature_importances = importances
            self._missing_artifacts = []
            self._ready = True

            logger.info(
                "Loaded XGBoost artifact version=%s (%d features, review=%.4f, block=%.4f)",
                self._model_version,
                len(self._feature_names),
                self._review_threshold,
                self._block_threshold,
            )
            return True

    def _reset_unready(self, missing: list[str]) -> None:
        self._ready = False
        self._model = None
        self._preprocessor = None
        self._feature_names = []
        self._model_version = None
        self._trained_at_utc = None
        self._validation_metrics = None
        self._pr_curve = None
        self._feature_importances = {}
        self._missing_artifacts = missing

    def _compute_and_update_causal_features(self, row: dict[str, Any]) -> dict[str, float | int | str]:
        """Compute strictly causal features matching Phase 1 engineering.py identically."""
        dt = float(row["TransactionDT"])
        amt = max(0.0, float(row["TransactionAmt"]))

        account_key, device_key, location_key = build_entity_keys_row(
            card1=row.get("card1"),
            addr1=row.get("addr1"),
            p_email=row.get("P_emaildomain"),
            device_info=row.get("DeviceInfo"),
            device_type=row.get("DeviceType"),
            addr2=row.get("addr2"),
        )

        state = self._account_states.setdefault(account_key, AccountCausalState())

        while state.window_1h and dt - state.window_1h[0] > 3600.0:
            state.window_1h.popleft()
        tx_count_1h = len(state.window_1h)

        while state.window_24h and dt - state.window_24h[0] > 86400.0:
            state.window_24h.popleft()
        tx_count_24h = len(state.window_24h)

        while state.window_7d and dt - state.window_7d[0] > 604800.0:
            state.window_7d.popleft()
        tx_count_7d = len(state.window_7d)

        prior_n = state.amt_count
        prior_mean = state.amt_mean
        prior_std = math.sqrt(state.amt_m2 / (prior_n - 1)) if prior_n >= 2 else 0.0

        if prior_n > 0 and state.last_dt is not None:
            amount_vs_user_avg = amt / max(prior_mean, 1.0)
            amount_zscore = (amt - prior_mean) / max(prior_std, 1.0)
            time_since_prev = max(0.0, dt - state.last_dt)
            prev_dev = state.last_device or "UNKNOWN|UNKNOWN"
            prev_loc = state.last_location or "UNKNOWN|UNKNOWN"
            device_change = (
                1
                if (
                    device_key != "UNKNOWN|UNKNOWN"
                    and prev_dev != "UNKNOWN|UNKNOWN"
                    and device_key != prev_dev
                )
                else 0
            )
            location_change = (
                1
                if (
                    location_key != "UNKNOWN|UNKNOWN"
                    and prev_loc != "UNKNOWN|UNKNOWN"
                    and location_key != prev_loc
                )
                else 0
            )
        else:
            amount_vs_user_avg = 1.0
            amount_zscore = 0.0
            time_since_prev = -1.0
            device_change = 0
            location_change = 0

        engineered: dict[str, float | int | str] = {
            "account_key": account_key,
            "device_key": device_key,
            "location_key": location_key,
            "tx_count_1h": tx_count_1h,
            "tx_count_24h": tx_count_24h,
            "tx_count_7d": tx_count_7d,
            "account_tx_count_prior": prior_n,
            "account_amt_mean_prior": round(prior_mean, 4),
            "account_amt_std_prior": round(prior_std, 4),
            "amount_vs_user_avg": round(amount_vs_user_avg, 4),
            "amount_zscore_deviation": round(amount_zscore, 4),
            "time_since_prev_tx_sec": round(time_since_prev, 2),
            "device_change_flag": device_change,
            "location_change_flag": location_change,
            "tx_hour_of_day": int((dt // 3600) % 24),
            "tx_day_of_week": int((dt // 86400) % 7),
            "amt_log1p": round(math.log1p(amt), 6),
            "amt_decimal_cents": round(amt - math.floor(amt), 4),
            "has_identity_record": int(row.get("has_identity_record", 0)),
        }

        # Advance state AFTER computing current transaction features
        state.window_1h.append(dt)
        state.window_24h.append(dt)
        state.window_7d.append(dt)
        new_n = prior_n + 1
        delta = amt - prior_mean
        new_mean = prior_mean + delta / new_n
        delta2 = amt - new_mean
        state.amt_count = new_n
        state.amt_mean = new_mean
        state.amt_m2 += delta * delta2
        state.last_dt = dt
        if device_key != "UNKNOWN|UNKNOWN" or state.last_device is None:
            state.last_device = device_key
        if location_key != "UNKNOWN|UNKNOWN" or state.last_location is None:
            state.last_location = location_key

        return engineered

    def _build_risk_indicators(
        self,
        engineered: dict[str, float | int | str],
        raw_row: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Derive transparent risk indicators from actual engineered values & model importances."""
        indicators: list[dict[str, Any]] = []
        candidates = [
            ("TransactionAmt", raw_row["TransactionAmt"], f"Amount ${float(raw_row['TransactionAmt']):,.2f}"),
            (
                "amount_vs_user_avg",
                engineered["amount_vs_user_avg"],
                f"{float(engineered['amount_vs_user_avg']):.2f}x prior account mean",
            ),
            ("tx_count_1h", engineered["tx_count_1h"], f"{int(engineered['tx_count_1h'])} prior tx in 1h"),
            (
                "device_change_flag",
                engineered["device_change_flag"],
                "Device changed vs. prior account tx" if engineered["device_change_flag"] else "Device unchanged",
            ),
            (
                "location_change_flag",
                engineered["location_change_flag"],
                "Billing location changed" if engineered["location_change_flag"] else "Billing location unchanged",
            ),
            (
                "time_since_prev_tx_sec",
                engineered["time_since_prev_tx_sec"],
                "First observed account tx"
                if float(engineered["time_since_prev_tx_sec"]) < 0
                else f"{float(engineered['time_since_prev_tx_sec']):.0f}s since previous tx",
            ),
        ]

        for feat_name, value, signal_text in candidates:
            indicators.append(
                {
                    "feature": feat_name,
                    "value": value,
                    "importance": self._feature_importances.get(feat_name),
                    "signal": signal_text,
                }
            )

        indicators.sort(key=lambda item: item["importance"] or 0.0, reverse=True)
        return indicators

    def _cache_features(
        self,
        transaction_id: str,
        engineered: dict[str, float | int | str],
        model_input_row: dict[str, float | int],
    ) -> None:
        self._feature_cache[transaction_id] = {
            "transaction_id": transaction_id,
            "model_version": self._model_version,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "engineered_features": engineered,
            "model_input_features": model_input_row,
        }
        self._feature_cache.move_to_end(transaction_id)
        while len(self._feature_cache) > self._max_feature_cache:
            self._feature_cache.popitem(last=False)

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run actual XGBoost predict_proba inference on a single validated transaction."""
        results = self.predict_batch([payload], record_batch_metric=False)
        return results[0]

    def predict_batch(
        self,
        payloads: list[dict[str, Any]],
        record_batch_metric: bool = True,
    ) -> list[dict[str, Any]]:
        """Run actual XGBoost predict_proba inference on a batch of transactions."""
        if not payloads:
            raise FeatureSchemaError("Batch must contain at least one transaction.")

        with self._lock:
            if not self._ready or self._model is None or self._preprocessor is None:
                self._error_count += 1
                raise ModelNotReadyError(
                    "Trained XGBoost model artifacts are not loaded. "
                    "Train Phase 1 first (`python -m src.models.train_xgboost`) and call POST /model/reload.",
                    missing_artifacts=list(self._missing_artifacts),
                )

            started = time.perf_counter()
            import pandas as pd

            raw_rows: list[dict[str, Any]] = []
            engineered_list: list[dict[str, float | int | str]] = []
            enriched_rows: list[dict[str, Any]] = []

            for item in payloads:
                raw_row = normalize_input_to_ieee_row(item)
                engineered = self._compute_and_update_causal_features(raw_row)
                merged = {**raw_row, **engineered}
                raw_rows.append(raw_row)
                engineered_list.append(engineered)
                enriched_rows.append(merged)

            frame = pd.DataFrame(enriched_rows)
            X_transformed = self._preprocessor.transform(frame)

            # Enforce strict training-serving column alignment
            actual_cols = list(X_transformed.columns)
            if actual_cols != self._feature_names:
                self._error_count += 1
                raise FeatureSchemaError(
                    f"Training-serving feature order mismatch: expected {len(self._feature_names)} "
                    f"columns, got {len(actual_cols)}."
                )

            proba_matrix = self._model.predict_proba(X_transformed)
            probs = [float(p[1]) for p in proba_matrix]

            total_elapsed_ms = (time.perf_counter() - started) * 1000.0
            per_tx_ms = round(total_elapsed_ms / len(payloads), 3)
            now_utc = datetime.now(timezone.utc).isoformat()

            if record_batch_metric:
                self._batch_requests += 1

            responses: list[dict[str, Any]] = []
            for idx, prob in enumerate(probs):
                prob_clamped = round(max(0.0, min(1.0, prob)), 6)
                decision = classify_decision(
                    prob_clamped,
                    review_threshold=self._review_threshold,
                    block_threshold=self._block_threshold,
                )

                self._total_predictions += 1
                self._decision_counts[decision] += 1
                self._latencies_ms.append(per_tx_ms)

                tx_id = str(raw_rows[idx]["TransactionID"])
                model_input_row = {
                    col: (
                        int(X_transformed.iloc[idx][col])
                        if str(X_transformed[col].dtype).startswith("int")
                        else round(float(X_transformed.iloc[idx][col]), 6)
                    )
                    for col in self._feature_names
                }
                self._cache_features(tx_id, engineered_list[idx], model_input_row)

                responses.append(
                    {
                        "transaction_id": tx_id,
                        "xgboost_probability": prob_clamped,
                        "gnn_probability": None,
                        "final_score": prob_clamped,
                        "decision": decision,
                        "inference_latency_ms": per_tx_ms,
                        "model_version": self._model_version,
                        "scoring_mode": "xgboost_only",
                        "gnn_available": False,
                        "thresholds": {
                            "review": self._review_threshold,
                            "block": self._block_threshold,
                        },
                        "risk_indicators": self._build_risk_indicators(
                            engineered_list[idx], raw_rows[idx]
                        ),
                        "scored_at_utc": now_utc,
                        "source": "xgboost_inference",
                    }
                )

            return responses

    def get_health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok" if self._ready else "degraded",
                "service": "sentinelrisk-ml-api",
                "model_ready": self._ready,
                "model_version": self._model_version,
                "missing_artifacts": list(self._missing_artifacts),
                "uptime_seconds": self.uptime_seconds(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

    def get_model_info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": self._ready,
                "model_type": "XGBoost Binary Classifier (Phase 1 artifact)",
                "model_version": self._model_version,
                "trained_at_utc": self._trained_at_utc,
                "artifact_dir": str(self.artifact_dir),
                "missing_artifacts": list(self._missing_artifacts),
                "feature_count": len(self._feature_names) if self._ready else None,
                "feature_names": list(self._feature_names),
                "thresholds": (
                    {"review": self._review_threshold, "block": self._block_threshold}
                    if self._ready
                    else None
                ),
                "validation_metrics": self._validation_metrics,
                "pr_curve": self._pr_curve,
                "gnn_status": "not_implemented_phase4",
            }

    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            lats = sorted(self._latencies_ms)
            if lats:
                avg_lat = round(sum(lats) / len(lats), 3)
                p95_lat = lats[min(len(lats) - 1, int(len(lats) * 0.95))]
                p99_lat = lats[min(len(lats) - 1, int(len(lats) * 0.99))]
            else:
                avg_lat = p95_lat = p99_lat = None

            return {
                "model_ready": self._ready,
                "model_version": self._model_version,
                "total_predictions": self._total_predictions,
                "batch_requests": self._batch_requests,
                "decision_counts": dict(self._decision_counts),
                "error_count": self._error_count,
                "avg_latency_ms": avg_lat,
                "p95_latency_ms": p95_lat,
                "p99_latency_ms": p99_lat,
                "cached_feature_records": len(self._feature_cache),
                "uptime_seconds": self.uptime_seconds(),
            }

    def get_transaction_features(self, transaction_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._feature_cache.get(transaction_id)
            if record is None:
                raise TransactionFeaturesNotFoundError(
                    f"No scored feature vector cached for transaction_id '{transaction_id}'."
                )
            return dict(record)
