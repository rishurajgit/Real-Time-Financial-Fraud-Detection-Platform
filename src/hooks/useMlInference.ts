import { useCallback, useEffect, useState } from 'react';
import type {
  MlApiError,
  MlHealthStatus,
  MlModelInfo,
  MlPredictRequest,
  MlPredictionResponse,
} from '../types/fraud';

export interface MlInferenceState {
  health: MlHealthStatus | null;
  modelInfo: MlModelInfo | null;
  serviceReachable: boolean;
  statusLoading: boolean;
  predicting: boolean;
  reloading: boolean;
  lastPrediction: MlPredictionResponse | null;
  realPredictions: MlPredictionResponse[];
  error: MlApiError | null;
}

async function parseError(res: Response): Promise<MlApiError> {
  try {
    const body = (await res.json()) as {
      error?: string;
      detail?: string;
      missing_artifacts?: string[];
    };
    return {
      status: res.status,
      error: body.error || `http_${res.status}`,
      detail: body.detail || `Request failed with status ${res.status}.`,
      missing_artifacts: body.missing_artifacts,
    };
  } catch {
    return {
      status: res.status,
      error: `http_${res.status}`,
      detail: `Request failed with status ${res.status}.`,
    };
  }
}

export function useMlInference() {
  const [state, setState] = useState<MlInferenceState>({
    health: null,
    modelInfo: null,
    serviceReachable: false,
    statusLoading: true,
    predicting: false,
    reloading: false,
    lastPrediction: null,
    realPredictions: [],
    error: null,
  });

  const refreshStatus = useCallback(async () => {
    setState((prev) => ({ ...prev, statusLoading: true }));
    try {
      const [healthRes, infoRes, predsRes] = await Promise.all([
        fetch('/api/v1/ml/health'),
        fetch('/api/v1/ml/model/info'),
        fetch('/api/v1/ml/predictions'),
      ]);

      const predsData = predsRes.ok
        ? ((await predsRes.json()) as { predictions: MlPredictionResponse[] }).predictions
        : [];

      if (!healthRes.ok) {
        const apiErr = await parseError(healthRes);
        setState((prev) => ({
          ...prev,
          health: null,
          modelInfo: null,
          serviceReachable: false,
          statusLoading: false,
          realPredictions: predsData,
          error: apiErr,
        }));
        return;
      }

      const health = (await healthRes.json()) as MlHealthStatus;
      const modelInfo = infoRes.ok ? ((await infoRes.json()) as MlModelInfo) : null;

      setState((prev) => ({
        ...prev,
        health,
        modelInfo,
        serviceReachable: true,
        statusLoading: false,
        realPredictions: predsData,
        error: null,
      }));
    } catch {
      setState((prev) => ({
        ...prev,
        health: null,
        modelInfo: null,
        serviceReachable: false,
        statusLoading: false,
        error: {
          status: 503,
          error: 'ml_service_unavailable',
          detail: 'Unable to reach Express /api/v1/ml status endpoints.',
        },
      }));
    }
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  const predictTransaction = useCallback(
    async (payload: MlPredictRequest): Promise<MlPredictionResponse | null> => {
      setState((prev) => ({ ...prev, predicting: true, error: null }));
      try {
        const res = await fetch('/api/v1/ml/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          const apiErr = await parseError(res);
          setState((prev) => ({
            ...prev,
            predicting: false,
            error: apiErr,
          }));
          return null;
        }

        const prediction = (await res.json()) as MlPredictionResponse;
        setState((prev) => ({
          ...prev,
          predicting: false,
          lastPrediction: prediction,
          realPredictions: [
            prediction,
            ...prev.realPredictions.filter((p) => p.transaction_id !== prediction.transaction_id),
          ].slice(0, 50),
          error: null,
        }));
        return prediction;
      } catch {
        setState((prev) => ({
          ...prev,
          predicting: false,
          error: {
            status: 503,
            error: 'network_error',
            detail: 'Network error while calling /api/v1/ml/predict.',
          },
        }));
        return null;
      }
    },
    []
  );

  const reloadModel = useCallback(async () => {
    setState((prev) => ({ ...prev, reloading: true, error: null }));
    try {
      const res = await fetch('/api/v1/ml/model/reload', { method: 'POST' });
      if (!res.ok) {
        const apiErr = await parseError(res);
        setState((prev) => ({ ...prev, reloading: false, error: apiErr }));
        return;
      }
      await refreshStatus();
      setState((prev) => ({ ...prev, reloading: false }));
    } catch {
      setState((prev) => ({
        ...prev,
        reloading: false,
        error: {
          status: 503,
          error: 'ml_service_unavailable',
          detail: 'Failed to contact /api/v1/ml/model/reload.',
        },
      }));
    }
  }, [refreshStatus]);

  return {
    ...state,
    refreshStatus,
    predictTransaction,
    reloadModel,
  };
}
