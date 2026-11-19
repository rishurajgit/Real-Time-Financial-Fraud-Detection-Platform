import assert from 'node:assert/strict';
import http from 'node:http';
import { after, before, describe, it } from 'node:test';
import express from 'express';
import { createMlProxyRouter } from '../src/server/mlProxy.ts';

type MockHandler = (req: http.IncomingMessage, body: string, res: http.ServerResponse) => void;

async function startHttpServer(
  listener: http.RequestListener
): Promise<{ server: http.Server; baseUrl: string }> {
  const server = http.createServer(listener);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') {
    throw new Error('Failed to bind ephemeral port.');
  }
  return { server, baseUrl: `http://127.0.0.1:${address.port}` };
}

describe('Express /api/v1/ml FastAPI Proxy Integration Tests', () => {
  let mockFastApi: http.Server;
  let mockFastApiUrl = '';
  let expressServer: http.Server;
  let expressUrl = '';
  let activeHandler: MockHandler;

  before(async () => {
    activeHandler = (_req, _body, res) => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok' }));
    };

    const fastApiStarted = await startHttpServer((req, res) => {
      let body = '';
      req.on('data', (chunk) => {
        body += chunk;
      });
      req.on('end', () => activeHandler(req, body, res));
    });
    mockFastApi = fastApiStarted.server;
    mockFastApiUrl = fastApiStarted.baseUrl;

    const app = express();
    app.use(express.json());
    app.use('/api/v1/ml', createMlProxyRouter({ baseUrl: mockFastApiUrl, timeoutMs: 250 }));
    const expressStarted = await startHttpServer(app);
    expressServer = expressStarted.server;
    expressUrl = expressStarted.baseUrl;
  });

  after(async () => {
    await new Promise<void>((resolve) => expressServer.close(() => resolve()));
    await new Promise<void>((resolve) => mockFastApi.close(() => resolve()));
  });

  it('proxies successful POST /api/v1/ml/predict and records real prediction', async () => {
    activeHandler = (req, body, res) => {
      assert.equal(req.url, '/predict');
      assert.equal(req.method, 'POST');
      const parsed = JSON.parse(body);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({
          transaction_id: parsed.transaction_id,
          xgboost_probability: 0.8742,
          gnn_probability: null,
          final_score: 0.8742,
          decision: 'Blocked',
          inference_latency_ms: 4.85,
          model_version: 'xgb-20261119-01',
          scoring_mode: 'xgboost_only',
          gnn_available: false,
          thresholds: { review: 0.5, block: 0.8 },
          risk_indicators: [],
          scored_at_utc: '2026-11-19T10:00:00Z',
          source: 'xgboost_inference',
        })
      );
    };

    const res = await fetch(`${expressUrl}/api/v1/ml/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        transaction_id: 'TX-PROXY-01',
        transaction_dt: 86400,
        transaction_amt: 920.5,
      }),
    });

    assert.equal(res.status, 200);
    const data = (await res.json()) as {
      transaction_id: string;
      decision: string;
      model_version: string;
      final_score: number;
    };
    assert.equal(data.transaction_id, 'TX-PROXY-01');
    assert.equal(data.decision, 'Blocked');
    assert.equal(data.model_version, 'xgb-20261119-01');
    assert.equal(data.final_score, 0.8742);

    // Verify stored in /api/v1/ml/predictions
    const listRes = await fetch(`${expressUrl}/api/v1/ml/predictions`);
    const listData = (await listRes.json()) as { count: number; predictions: Array<{ transaction_id: string }> };
    assert.equal(listData.count, 1);
    assert.equal(listData.predictions[0].transaction_id, 'TX-PROXY-01');
  });

  it('proxies POST /api/v1/ml/predict/batch successfully', async () => {
    activeHandler = (req, body, res) => {
      assert.equal(req.url, '/predict/batch');
      const parsed = JSON.parse(body) as { transactions: Array<{ transaction_id: string }> };
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({
          count: parsed.transactions.length,
          total_latency_ms: 6.1,
          model_version: 'xgb-20261119-01',
          predictions: parsed.transactions.map((t) => ({
            transaction_id: t.transaction_id,
            xgboost_probability: 0.12,
            gnn_probability: null,
            final_score: 0.12,
            decision: 'Approved',
            inference_latency_ms: 3.05,
            model_version: 'xgb-20261119-01',
            scoring_mode: 'xgboost_only',
            gnn_available: false,
            thresholds: { review: 0.5, block: 0.8 },
            risk_indicators: [],
            scored_at_utc: '2026-11-19T10:00:00Z',
            source: 'xgboost_inference',
          })),
        })
      );
    };

    const res = await fetch(`${expressUrl}/api/v1/ml/predict/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        transactions: [
          { transaction_id: 'TX-B1', transaction_dt: 100, transaction_amt: 20 },
          { transaction_id: 'TX-B2', transaction_dt: 200, transaction_amt: 35 },
        ],
      }),
    });
    assert.equal(res.status, 200);
    const data = (await res.json()) as { count: number };
    assert.equal(data.count, 2);
  });

  it('rejects invalid transaction payloads with HTTP 400 before calling FastAPI', async () => {
    let upstreamCalled = false;
    activeHandler = (_req, _body, res) => {
      upstreamCalled = true;
      res.writeHead(200);
      res.end();
    };

    const res = await fetch(`${expressUrl}/api/v1/ml/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transaction_id: 'TX-INVALID', transaction_dt: 100, transaction_amt: -50 }),
    });
    assert.equal(res.status, 400);
    const body = (await res.json()) as { error: string };
    assert.equal(body.error, 'invalid_payload');
    assert.equal(upstreamCalled, false);
  });

  it('propagates FastAPI 503 model_not_ready without leaking host filesystem paths', async () => {
    activeHandler = (_req, _body, res) => {
      res.writeHead(503, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({
          error: 'model_not_ready',
          detail: 'Trained XGBoost model artifacts are not loaded.',
          missing_artifacts: ['/internal/secret/path/artifacts/models/xgboost/model_bundle.joblib'],
        })
      );
    };

    const res = await fetch(`${expressUrl}/api/v1/ml/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transaction_id: 'TX-503', transaction_dt: 100, transaction_amt: 50 }),
    });
    assert.equal(res.status, 503);
    const body = (await res.json()) as { error: string; missing_artifacts: string[] };
    assert.equal(body.error, 'model_not_ready');
    assert.deepEqual(body.missing_artifacts, ['model_bundle.joblib']);
  });

  it('returns HTTP 504 on FastAPI timeout without exposing ML_SERVICE_URL', async () => {
    activeHandler = (_req, _body, res) => {
      setTimeout(() => {
        if (!res.writableEnded) {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end('{}');
        }
      }, 600);
    };

    const res = await fetch(`${expressUrl}/api/v1/ml/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transaction_id: 'TX-SLOW', transaction_dt: 100, transaction_amt: 50 }),
    });
    assert.equal(res.status, 504);
    const body = (await res.json()) as { error: string; detail: string };
    assert.equal(body.error, 'ml_service_timeout');
    assert.equal(body.detail.includes('127.0.0.1'), false);
  });

  it('returns HTTP 503 ml_service_unavailable when FastAPI is unreachable (no fake fallback)', async () => {
    const offlineApp = express();
    offlineApp.use(express.json());
    // Port 1 is closed / unreachable
    offlineApp.use('/api/v1/ml', createMlProxyRouter({ baseUrl: 'http://127.0.0.1:1', timeoutMs: 250 }));
    const started = await startHttpServer(offlineApp);

    try {
      const res = await fetch(`${started.baseUrl}/api/v1/ml/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transaction_id: 'TX-OFFLINE', transaction_dt: 100, transaction_amt: 50 }),
      });
      assert.equal(res.status, 503);
      const body = (await res.json()) as { error: string; detail: string; final_score?: number };
      assert.equal(body.error, 'ml_service_unavailable');
      assert.equal(body.final_score, undefined);
      assert.equal(body.detail.includes('127.0.0.1:1'), false);
    } finally {
      await new Promise<void>((resolve) => started.server.close(() => resolve()));
    }
  });

  it('verifies sentinelrisk-ml-api identity on GET /api/v1/ml/health', async () => {
    activeHandler = (_req, _body, res) => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'healthy' }));
    };
    const wrongService = await fetch(`${expressUrl}/api/v1/ml/health`);
    assert.equal(wrongService.status, 503);

    activeHandler = (_req, _body, res) => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({
          status: 'ok',
          service: 'sentinelrisk-ml-api',
          model_ready: true,
          model_version: 'xgb-20261119-01',
          missing_artifacts: [],
          uptime_seconds: 5,
          timestamp_utc: '2026-11-19T12:00:00Z',
        })
      );
    };
    const rightService = await fetch(`${expressUrl}/api/v1/ml/health`);
    assert.equal(rightService.status, 200);
    const data = (await rightService.json()) as { model_ready: boolean };
    assert.equal(data.model_ready, true);
  });
});
