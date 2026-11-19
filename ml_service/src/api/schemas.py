"""Pydantic request and response schemas for the SentinelRisk FastAPI inference service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionInput(BaseModel):
    """Input transaction payload aligned with IEEE-CIS raw fields and optional precomputed attributes."""

    model_config = ConfigDict(extra="allow")

    transaction_id: str = Field(..., min_length=1, max_length=128, description="Unique transaction identifier.")
    transaction_dt: float = Field(..., ge=0.0, description="Transaction timestamp in seconds (IEEE-CIS TransactionDT).")
    transaction_amt: float = Field(..., ge=0.0, description="Transaction monetary amount in USD.")
    product_cd: str | None = Field(default="W", description="IEEE-CIS ProductCD (W, H, C, S, R).")
    card1: float | None = Field(default=None, description="Primary payment card identifier.")
    card2: float | None = Field(default=None)
    card3: float | None = Field(default=None)
    card4: str | None = Field(default=None, description="Card network (visa, mastercard, american express, discover).")
    card5: float | None = Field(default=None)
    card6: str | None = Field(default=None, description="Card type (debit, credit).")
    addr1: float | None = Field(default=None, description="Billing region identifier.")
    addr2: float | None = Field(default=None, description="Billing country identifier.")
    dist1: float | None = Field(default=None, ge=0.0, description="Distance metric 1.")
    dist2: float | None = Field(default=None, ge=0.0, description="Distance metric 2.")
    p_emaildomain: str | None = Field(default=None, description="Purchaser email domain.")
    r_emaildomain: str | None = Field(default=None, description="Recipient email domain.")
    device_type: str | None = Field(default=None, description="Device type (desktop, mobile).")
    device_info: str | None = Field(default=None, description="Device fingerprint / user-agent summary.")
    has_identity_record: int | None = Field(default=None, ge=0, le=1)
    extra_features: dict[str, float | int | str | None] = Field(
        default_factory=dict,
        description="Optional additional IEEE-CIS C/D/M/V/id_* columns.",
    )

    @field_validator("transaction_id")
    @classmethod
    def _strip_transaction_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("transaction_id must not be blank.")
        return cleaned


class BatchPredictRequest(BaseModel):
    """Batch prediction payload."""

    transactions: list[TransactionInput] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Non-empty list of transactions (up to 500 per batch).",
    )


class RiskIndicator(BaseModel):
    feature: str
    value: float | int | str
    importance: float | None = None
    signal: str


class PredictionResponse(BaseModel):
    transaction_id: str
    xgboost_probability: float = Field(..., ge=0.0, le=1.0)
    gnn_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    final_score: float = Field(..., ge=0.0, le=1.0)
    decision: Literal["Approved", "Review", "Blocked"]
    inference_latency_ms: float = Field(..., ge=0.0)
    model_version: str
    scoring_mode: Literal["xgboost_only", "hybrid_xgb_gnn"] = "xgboost_only"
    gnn_available: bool = False
    thresholds: dict[str, float]
    risk_indicators: list[RiskIndicator] = Field(default_factory=list)
    scored_at_utc: str
    source: Literal["xgboost_inference"] = "xgboost_inference"


class BatchPredictionResponse(BaseModel):
    count: int
    total_latency_ms: float
    model_version: str
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    model_ready: bool
    model_version: str | None
    missing_artifacts: list[str]
    uptime_seconds: float
    timestamp_utc: str


class ModelInfoResponse(BaseModel):
    ready: bool
    model_type: str
    model_version: str | None
    trained_at_utc: str | None
    artifact_dir: str
    missing_artifacts: list[str]
    feature_count: int | None
    feature_names: list[str]
    thresholds: dict[str, float] | None
    validation_metrics: dict[str, Any] | None
    pr_curve: list[dict[str, float]] | None
    gnn_status: Literal["not_implemented_phase4", "unavailable", "ready"] = "not_implemented_phase4"


class MetricsResponse(BaseModel):
    model_ready: bool
    model_version: str | None
    total_predictions: int
    batch_requests: int
    decision_counts: dict[str, int]
    error_count: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None
    cached_feature_records: int
    uptime_seconds: float


class ModelReloadResponse(BaseModel):
    reloaded: bool
    ready: bool
    model_version: str | None
    missing_artifacts: list[str]
    message: str


class FeatureLookupResponse(BaseModel):
    transaction_id: str
    model_version: str
    captured_at_utc: str
    engineered_features: dict[str, float | int | str]
    model_input_features: dict[str, float | int]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    missing_artifacts: list[str] | None = None
    timestamp_utc: str
