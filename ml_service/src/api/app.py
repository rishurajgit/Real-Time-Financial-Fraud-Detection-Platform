"""SentinelRisk FastAPI ML Inference Service (Phase 2).

Endpoints:
  - GET  /health
  - GET  /model/info
  - POST /predict
  - POST /predict/batch
  - GET  /metrics
  - POST /model/reload
  - GET  /features/{transaction_id}

Run locally:
    cd ml_service
    uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.model_service import (
    FeatureSchemaError,
    ModelInferenceService,
    ModelNotReadyError,
    TransactionFeaturesNotFoundError,
)
from src.api.schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    ErrorResponse,
    FeatureLookupResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    ModelReloadResponse,
    PredictionResponse,
    TransactionInput,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sentinelrisk.api")


def _error_payload(
    error: str,
    detail: str,
    missing_artifacts: list[str] | None = None,
) -> dict[str, object]:
    return ErrorResponse(
        error=error,
        detail=detail,
        missing_artifacts=missing_artifacts,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    ).model_dump()


def create_app(
    artifact_dir: Path | None = None,
    service_instance: ModelInferenceService | None = None,
) -> FastAPI:
    """Create configured FastAPI application with lifespan artifact loading and error handlers."""
    service = service_instance or ModelInferenceService(artifact_dir=artifact_dir)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info("Starting SentinelRisk FastAPI ML Service...")
        loaded = service.load_artifacts()
        if loaded:
            logger.info("Model artifacts loaded and ready for inference.")
        else:
            logger.warning(
                "FastAPI started in degraded (model-not-ready) mode. Missing artifacts: %s",
                service.missing_artifacts,
            )
        yield
        logger.info("Shutting down SentinelRisk FastAPI ML Service.")

    app = FastAPI(
        title="SentinelRisk ML Inference API",
        description=(
            "Real-time XGBoost financial fraud detection inference service built on "
            "Phase 1 IEEE-CIS artifacts and causal behavioral features."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.inference_service = service

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "%s %s -> %d (%.2f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.exception_handler(ModelNotReadyError)
    async def handle_model_not_ready(_req: Request, exc: ModelNotReadyError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_payload(
                error="model_not_ready",
                detail=str(exc),
                missing_artifacts=exc.missing_artifacts,
            ),
        )

    @app.exception_handler(FeatureSchemaError)
    async def handle_feature_schema(_req: Request, exc: FeatureSchemaError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_payload(error="invalid_feature_schema", detail=str(exc)),
        )

    @app.exception_handler(TransactionFeaturesNotFoundError)
    async def handle_features_not_found(
        _req: Request, exc: TransactionFeaturesNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=_error_payload(error="features_not_found", detail=str(exc)),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_req: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_payload(
                error="validation_error",
                detail=str(exc.errors()),
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_req: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled inference server error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload(
                error="internal_server_error",
                detail="Unexpected error during ML inference.",
            ),
        )

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        return HealthResponse(**service.get_health())

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
    async def model_info() -> ModelInfoResponse:
        return ModelInfoResponse(**service.get_model_info())

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        responses={503: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
        tags=["Inference"],
    )
    async def predict(payload: TransactionInput) -> PredictionResponse:
        result = service.predict_one(payload.model_dump())
        return PredictionResponse(**result)

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        responses={503: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
        tags=["Inference"],
    )
    async def predict_batch(request: BatchPredictRequest) -> BatchPredictionResponse:
        started = time.perf_counter()
        raw_items = [item.model_dump() for item in request.transactions]
        predictions = service.predict_batch(raw_items, record_batch_metric=True)
        total_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return BatchPredictionResponse(
            count=len(predictions),
            total_latency_ms=total_ms,
            model_version=predictions[0]["model_version"],
            predictions=[PredictionResponse(**p) for p in predictions],
        )

    @app.get("/metrics", response_model=MetricsResponse, tags=["System"])
    async def metrics() -> MetricsResponse:
        return MetricsResponse(**service.get_metrics())

    @app.post("/model/reload", response_model=ModelReloadResponse, tags=["Model"])
    async def reload_model() -> ModelReloadResponse:
        ok = service.load_artifacts()
        info = service.get_model_info()
        return ModelReloadResponse(
            reloaded=ok,
            ready=info["ready"],
            model_version=info["model_version"],
            missing_artifacts=info["missing_artifacts"],
            message=(
                f"Loaded model version {info['model_version']}."
                if ok
                else "Model artifacts still unavailable; service remains in model-not-ready state."
            ),
        )

    @app.get(
        "/features/{transaction_id}",
        response_model=FeatureLookupResponse,
        responses={404: {"model": ErrorResponse}},
        tags=["Features"],
    )
    async def get_features(transaction_id: str) -> FeatureLookupResponse:
        record = service.get_transaction_features(transaction_id)
        return FeatureLookupResponse(**record)

    return app


app = create_app()
