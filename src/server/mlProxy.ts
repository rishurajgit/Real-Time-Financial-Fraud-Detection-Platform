import { Router, type Request, type Response } from 'express';

export interface MlProxyConfig {
  baseUrl?: string;
  timeoutMs?: number;
}

export interface MlPredictionRecord {
  transaction_id: string;
  xgboost_probability: number;
  gnn_probability: number | null;
  final_score: number;
  decision: 'Approved' | 'Review' | 'Blocked';
  inference_latency_ms: number;
  model_version: string;
  scoring_mode: 'xgboost_only' | 'hybrid_xgb_gnn';
  gnn_available: boolean;
  thresholds: { review: number; block: number };
  risk_indicators: Array<{
    feature: string;
    value: number | string;
    importance: number | null;
    signal: string;
  }>;
  scored_at_utc: string;
  source: 'xgboost_inference';
}

interface ValidationResult {
  valid: boolean;
  error?: string;
}

export function validateMlTransactionPayload(body: unknown): ValidationResult {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    return { valid: false, error: 'Request body must be a JSON object.' };
  }
  const obj = body as Record<string, unknown>;
  if (typeof obj.transaction_id !== 'string' || obj.transaction_id.trim().length === 0) {
    return { valid: false, error: 'Field "transaction_id" must be a non-empty string.' };
  }
  if (
    typeof obj.transaction_dt !== 'number' ||
    !Number.isFinite(obj.transaction_dt) ||
    obj.transaction_dt < 0
  ) {
    return { valid: false, error: 'Field "transaction_dt" must be a finite non-negative number.' };
  }
  if (
    typeof obj.transaction_amt !== 'number' ||
    !Number.isFinite(obj.transaction_amt) ||
    obj.transaction_amt < 0
  ) {
    return { valid: false, error: 'Field "transaction_amt" must be a finite non-negative number.' };
  }
  if (
    obj.extra_features !== undefined &&
    (typeof obj.extra_features !== 'object' ||
      obj.extra_features === null ||
      Array.isArray(obj.extra_features))
  ) {
    return { valid: false, error: 'Field "extra_features" must be a JSON object when provided.' };
  }
  return { valid: true };
}

function sanitizeMissingArtifacts(raw: unknown): string[] | undefined {
  if (!Array.isArray(raw)) return undefined;
  return raw.map((item) => {
    const text = String(item);
    // Avoid leaking full host directory paths; retain filename and concise reason
    const normalized = text.replace(/\\/g, '/');
    const parts = normalized.split('/');
    return parts[parts.length - 1] || text;
  });
}

function sanitizeUpstreamError(status: number, body: unknown): Record<string, unknown> {
  if (body && typeof body === 'object' && !Array.isArray(body)) {
    const obj = body as Record<string, unknown>;
    const sanitized: Record<string, unknown> = {
      error: typeof obj.error === 'string' ? obj.error : `ml_upstream_http_${status}`,
      detail:
        typeof obj.detail === 'string'
          ? obj.detail
          : 'ML inference service returned an error response.',
    };
    const missing = sanitizeMissingArtifacts(obj.missing_artifacts);
    if (missing) {
      sanitized.missing_artifacts = missing;
    }
    return sanitized;
  }
  return {
    error: `ml_upstream_http_${status}`,
    detail: 'ML inference service returned an unexpected error response.',
  };
}

export function createMlProxyRouter(config: MlProxyConfig = {}): Router {
  const router = Router();
  const recentRealPredictions: MlPredictionRecord[] = [];
  const maxStoredPredictions = 100;

  const getBaseUrl = (): string => {
    const raw = config.baseUrl ?? process.env.ML_SERVICE_URL ?? 'http://127.0.0.1:8000';
    return raw.replace(/\/+$/, '');
  };

  const getTimeoutMs = (): number => {
    const envVal = process.env.ML_SERVICE_TIMEOUT_MS
      ? Number(process.env.ML_SERVICE_TIMEOUT_MS)
      : undefined;
    const resolved = config.timeoutMs ?? envVal ?? 5000;
    return Number.isFinite(resolved) && resolved > 0 ? resolved : 5000;
  };

  async function forwardToFastApi(
    res: Response,
    path: string,
    method: 'GET' | 'POST',
    payload?: unknown,
    onSuccess?: (data: unknown) => void
  ): Promise<void> {
    const controller = new AbortController();
    const timeoutMs = getTimeoutMs();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const upstreamRes = await fetch(`${getBaseUrl()}${path}`, {
        method,
        headers: payload !== undefined ? { 'Content-Type': 'application/json' } : undefined,
        body: payload !== undefined ? JSON.stringify(payload) : undefined,
        signal: controller.signal,
      });

      const contentType = upstreamRes.headers.get('content-type') || '';
      const parsedBody: unknown = contentType.includes('application/json')
        ? await upstreamRes.json().catch(() => null)
        : await upstreamRes.text().catch(() => null);

      if (!upstreamRes.ok) {
        res.status(upstreamRes.status).json(sanitizeUpstreamError(upstreamRes.status, parsedBody));
        return;
      }

      if (onSuccess) {
        onSuccess(parsedBody);
      }
      res.status(upstreamRes.status).json(parsedBody);
    } catch (err) {
      const isAbort =
        err instanceof Error && (err.name === 'AbortError' || err.message.includes('aborted'));
      if (isAbort) {
        res.status(504).json({
          error: 'ml_service_timeout',
          detail: 'ML inference service timed out before returning a prediction.',
        });
        return;
      }

      res.status(503).json({
        error: 'ml_service_unavailable',
        detail:
          'ML inference service is unreachable. Start the FastAPI service and verify connectivity.',
      });
    } finally {
      clearTimeout(timer);
    }
  }

  router.get('/health', async (_req: Request, res: Response) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), getTimeoutMs());
    try {
      const upstreamRes = await fetch(`${getBaseUrl()}/health`, {
        method: 'GET',
        signal: controller.signal,
      });
      const parsed: unknown = await upstreamRes.json().catch(() => null);
      if (
        !upstreamRes.ok ||
        !parsed ||
        typeof parsed !== 'object' ||
        (parsed as { service?: unknown }).service !== 'sentinelrisk-ml-api'
      ) {
        res.status(503).json({
          error: 'ml_service_unavailable',
          detail:
            'SentinelRisk FastAPI ML service is not running at the configured ML_SERVICE_URL.',
        });
        return;
      }
      res.status(200).json(parsed);
    } catch (err) {
      const isAbort =
        err instanceof Error && (err.name === 'AbortError' || err.message.includes('aborted'));
      res.status(isAbort ? 504 : 503).json({
        error: isAbort ? 'ml_service_timeout' : 'ml_service_unavailable',
        detail: isAbort
          ? 'ML inference service timed out during health check.'
          : 'ML inference service is unreachable. Start the FastAPI service and verify connectivity.',
      });
    } finally {
      clearTimeout(timer);
    }
  });

  router.get('/model/info', async (_req: Request, res: Response) => {
    await forwardToFastApi(res, '/model/info', 'GET');
  });

  router.get('/metrics', async (_req: Request, res: Response) => {
    await forwardToFastApi(res, '/metrics', 'GET');
  });

  router.post('/model/reload', async (_req: Request, res: Response) => {
    await forwardToFastApi(res, '/model/reload', 'POST');
  });

  router.get('/features/:transactionId', async (req: Request, res: Response) => {
    const txId = req.params.transactionId?.trim();
    if (!txId) {
      res.status(400).json({ error: 'invalid_request', detail: 'transactionId is required.' });
      return;
    }
    await forwardToFastApi(res, `/features/${encodeURIComponent(txId)}`, 'GET');
  });

  router.get('/predictions', (_req: Request, res: Response) => {
    res.json({
      count: recentRealPredictions.length,
      predictions: recentRealPredictions,
    });
  });

  router.post('/predict', async (req: Request, res: Response) => {
    const check = validateMlTransactionPayload(req.body);
    if (!check.valid) {
      res.status(400).json({ error: 'invalid_payload', detail: check.error });
      return;
    }

    await forwardToFastApi(res, '/predict', 'POST', req.body, (data) => {
      if (data && typeof data === 'object' && 'transaction_id' in data) {
        recentRealPredictions.unshift(data as MlPredictionRecord);
        if (recentRealPredictions.length > maxStoredPredictions) {
          recentRealPredictions.pop();
        }
      }
    });
  });

  router.post('/predict/batch', async (req: Request, res: Response) => {
    const transactions = req.body?.transactions;
    if (!Array.isArray(transactions) || transactions.length === 0 || transactions.length > 500) {
      res.status(400).json({
        error: 'invalid_payload',
        detail: 'Field "transactions" must be a non-empty array of 1 to 500 transaction objects.',
      });
      return;
    }

    for (let i = 0; i < transactions.length; i++) {
      const itemCheck = validateMlTransactionPayload(transactions[i]);
      if (!itemCheck.valid) {
        res.status(400).json({
          error: 'invalid_payload',
          detail: `Transaction at index ${i}: ${itemCheck.error}`,
        });
        return;
      }
    }

    await forwardToFastApi(res, '/predict/batch', 'POST', { transactions }, (data) => {
      if (
        data &&
        typeof data === 'object' &&
        Array.isArray((data as { predictions?: unknown[] }).predictions)
      ) {
        const preds = (data as { predictions: MlPredictionRecord[] }).predictions;
        for (let i = preds.length - 1; i >= 0; i--) {
          recentRealPredictions.unshift(preds[i]);
        }
        if (recentRealPredictions.length > maxStoredPredictions) {
          recentRealPredictions.length = maxStoredPredictions;
        }
      }
    });
  });

  return router;
}
