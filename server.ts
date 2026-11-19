import 'dotenv/config';
import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import { fraudPlatform } from './src/server/fraudPlatform.ts';
import { createMlProxyRouter } from './src/server/mlProxy.ts';
import type { CaseStatus, FeedbackLabel, ScoreTransactionRequest } from './src/types/fraud.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const port = Number(process.env.PORT || 3000);

  app.use(express.json());

  // Phase 2: Real FastAPI ML Inference Service proxy (no simulated fallbacks)
  app.use('/api/v1/ml', createMlProxyRouter());

  // Health probe
  app.get('/api/health', (_req, res) => {
    res.json({ ok: true, service: 'sentinel-risk-api', timestamp: new Date().toISOString() });
  });

  // Full initial state snapshot
  app.get('/api/v1/state', (_req, res) => {
    res.json(fraudPlatform.getSnapshot());
  });

  // Real-time Server-Sent Events (SSE) Kafka consumer feed
  app.get('/api/v1/stream', (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const sendEvent = (event: string, payload: unknown) => {
      res.write(`event: ${event}\n`);
      res.write(`data: ${JSON.stringify(payload)}\n\n`);
    };

    const unsubscribe = fraudPlatform.subscribe(sendEvent);
    const keepAlive = setInterval(() => {
      res.write(': keepalive\n\n');
    }, 20000);

    req.on('close', () => {
      clearInterval(keepAlive);
      unsubscribe();
    });
  });

  // FastAPI-equivalent real-time scoring endpoint
  app.post('/api/v1/score', (req, res) => {
    const body = req.body as Partial<ScoreTransactionRequest>;
    if (!body || typeof body.amount !== 'number' || !body.accountId) {
      res.status(400).json({ error: 'accountId and numeric amount are required.' });
      return;
    }

    const scored = fraudPlatform.scorePayload({
      accountId: String(body.accountId),
      amount: Number(body.amount),
      merchantName: body.merchantName || 'Apex Electronics',
      merchantCategory: body.merchantCategory || 'Consumer Tech',
      location: body.location || 'New York, US',
      deviceId: body.deviceId || 'DEV-F991A',
      cardHash: body.cardHash || 'CARD-****-8841',
      ipAddress: body.ipAddress || '185.220.101.44',
      txType: body.txType || 'CARD_NOT_PRESENT',
      txCount1h: typeof body.txCount1h === 'number' ? body.txCount1h : undefined,
      geoDistanceKm: typeof body.geoDistanceKm === 'number' ? body.geoDistanceKm : undefined,
      deviceChangeFlag: body.deviceChangeFlag === 1 ? 1 : body.deviceChangeFlag === 0 ? 0 : undefined,
      ingestIntoStream: body.ingestIntoStream !== false,
    });

    res.json(scored);
  });

  // Stream replay controls
  app.post('/api/v1/stream/control', (req, res) => {
    const updated = fraudPlatform.updateStreamControl(req.body || {});
    res.json(updated);
  });

  // Inject coordinated fraud ring burst
  app.post('/api/v1/stream/burst', (_req, res) => {
    const burst = fraudPlatform.injectFraudBurst();
    res.json({ count: burst.length, transactions: burst });
  });

  // Analyst case status and feedback labeling
  app.patch('/api/v1/cases/:id', (req, res) => {
    const { status, feedbackLabel, notes } = (req.body || {}) as {
      status?: CaseStatus;
      feedbackLabel?: FeedbackLabel;
      notes?: string;
    };
    const updated = fraudPlatform.updateCase(req.params.id, { status, feedbackLabel, notes });
    if (!updated) {
      res.status(404).json({ error: 'Case not found.' });
      return;
    }
    res.json(updated);
  });

  // Fraud ring enforcement actions (freeze / escalate / release)
  app.post('/api/v1/rings/:id/action', (req, res) => {
    const action = req.body?.action as 'freeze' | 'escalate' | 'release';
    if (!['freeze', 'escalate', 'release'].includes(action)) {
      res.status(400).json({ error: 'Valid action required: freeze, escalate, or release.' });
      return;
    }
    const updated = fraudPlatform.performRingAction(req.params.id, action);
    if (!updated) {
      res.status(404).json({ error: 'Fraud ring not found.' });
      return;
    }
    res.json(updated);
  });

  // Model blend weight, threshold, and MLflow production run promotion
  app.post('/api/v1/model/config', (req, res) => {
    const updated = fraudPlatform.updateModelConfig(req.body || {});
    res.json(updated);
  });

  // Drift monitor settings (auto-retrain & PSI threshold)
  app.post('/api/v1/drift/config', (req, res) => {
    const report = fraudPlatform.setDriftSettings(req.body || {});
    res.json(report);
  });

  // Trigger automated retraining job
  app.post('/api/v1/retrain', (req, res) => {
    const reason = typeof req.body?.reason === 'string' ? req.body.reason : undefined;
    const job = fraudPlatform.triggerRetraining(reason);
    res.json(job);
  });

  const isProd = process.env.NODE_ENV === 'production' && fs.existsSync(path.resolve(__dirname, 'dist'));

  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(port, '0.0.0.0', () => {
    console.log(`SentinelRisk server listening on http://0.0.0.0:${port}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start SentinelRisk server:', err);
  process.exit(1);
});
