import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { RealMlInferencePanel } from '../src/components/RealMlInferencePanel.tsx';
import { ModelMlflowView } from '../src/components/ModelMlflowView.tsx';
import type { MlPredictionResponse } from '../src/types/fraud.ts';

const noopAsync = async () => {};
const noopPredict = async () => null;

const SAMPLE_PREDICTION: MlPredictionResponse = {
  transaction_id: 'IEEE-99812',
  xgboost_probability: 0.8742,
  gnn_probability: null,
  final_score: 0.8742,
  decision: 'Blocked',
  inference_latency_ms: 4.85,
  model_version: 'xgb-20261119-01',
  scoring_mode: 'xgboost_only',
  gnn_available: false,
  thresholds: { review: 0.5, block: 0.8 },
  risk_indicators: [
    {
      feature: 'TransactionAmt',
      value: 1850,
      importance: 0.34,
      signal: 'Amount $1,850.00',
    },
  ],
  scored_at_utc: '2026-11-19T12:00:00Z',
  source: 'xgboost_inference',
};

describe('Frontend RealMlInferencePanel & ModelMlflowView Integration Tests', () => {
  it('renders clear loading state during active prediction request', () => {
    const html = renderToStaticMarkup(
      React.createElement(RealMlInferencePanel, {
        health: {
          status: 'ok',
          service: 'sentinelrisk-ml-api',
          model_ready: true,
          model_version: 'xgb-20261119-01',
          missing_artifacts: [],
          uptime_seconds: 12,
          timestamp_utc: '2026-11-19T12:00:00Z',
        },
        modelInfo: null,
        serviceReachable: true,
        statusLoading: false,
        predicting: true,
        reloading: false,
        lastPrediction: null,
        realPredictions: [],
        error: null,
        onPredict: noopPredict,
        onRefreshStatus: noopAsync,
        onReloadModel: noopAsync,
      })
    );

    assert.match(html, /Scoring via XGBoost\.\.\./);
    assert.match(html, /Running real XGBoost predict_proba\(\) inference\.\.\./);
  });

  it('renders successful real XGBoost prediction with probability, score, decision, version, and latency', () => {
    const html = renderToStaticMarkup(
      React.createElement(RealMlInferencePanel, {
        health: {
          status: 'ok',
          service: 'sentinelrisk-ml-api',
          model_ready: true,
          model_version: 'xgb-20261119-01',
          missing_artifacts: [],
          uptime_seconds: 42,
          timestamp_utc: '2026-11-19T12:00:00Z',
        },
        modelInfo: {
          ready: true,
          model_type: 'XGBoost Binary Classifier',
          model_version: 'xgb-20261119-01',
          trained_at_utc: '2026-11-19T11:00:00Z',
          artifact_dir: 'artifacts/models/xgboost',
          missing_artifacts: [],
          feature_count: 54,
          feature_names: [],
          thresholds: { review: 0.5, block: 0.8 },
          validation_metrics: { pr_auc: 0.8124, roc_auc: 0.9432, f1_score: 0.742 },
          pr_curve: null,
          gnn_status: 'not_implemented_phase4',
        },
        serviceReachable: true,
        statusLoading: false,
        predicting: false,
        reloading: false,
        lastPrediction: SAMPLE_PREDICTION,
        realPredictions: [SAMPLE_PREDICTION],
        error: null,
        onPredict: noopPredict,
        onRefreshStatus: noopAsync,
        onReloadModel: noopAsync,
      })
    );

    assert.match(html, /Model Ready/);
    assert.match(html, /IEEE-99812/);
    assert.match(html, /87\.42%/);
    assert.match(html, /0\.8742/);
    assert.match(html, /Blocked/);
    assert.match(html, /xgb-20261119-01/);
    assert.match(html, /4\.85 ms/);
    assert.match(html, /Real Model-Scored Transactions \(1\)/);
  });

  it('renders model unavailable state (HTTP 503) showing missing artifact without fake predictions', () => {
    const html = renderToStaticMarkup(
      React.createElement(RealMlInferencePanel, {
        health: {
          status: 'degraded',
          service: 'sentinelrisk-ml-api',
          model_ready: false,
          model_version: null,
          missing_artifacts: ['model_bundle.joblib'],
          uptime_seconds: 10,
          timestamp_utc: '2026-11-19T12:00:00Z',
        },
        modelInfo: {
          ready: false,
          model_type: 'XGBoost Binary Classifier',
          model_version: null,
          trained_at_utc: null,
          artifact_dir: 'artifacts/models/xgboost',
          missing_artifacts: ['model_bundle.joblib'],
          feature_count: null,
          feature_names: [],
          thresholds: null,
          validation_metrics: null,
          pr_curve: null,
          gnn_status: 'not_implemented_phase4',
        },
        serviceReachable: true,
        statusLoading: false,
        predicting: false,
        reloading: false,
        lastPrediction: null,
        realPredictions: [],
        error: null,
        onPredict: noopPredict,
        onRefreshStatus: noopAsync,
        onReloadModel: noopAsync,
      })
    );

    assert.match(html, /Model Unavailable \(HTTP 503\)/);
    assert.match(html, /model_bundle\.joblib/);
    assert.doesNotMatch(html, /Fraud Probability/);
    assert.doesNotMatch(html, /Real Model-Scored Transactions/);
  });

  it('renders API error state cleanly on timeout or validation failure', () => {
    const html = renderToStaticMarkup(
      React.createElement(RealMlInferencePanel, {
        health: null,
        modelInfo: null,
        serviceReachable: false,
        statusLoading: false,
        predicting: false,
        reloading: false,
        lastPrediction: null,
        realPredictions: [],
        error: {
          status: 504,
          error: 'ml_service_timeout',
          detail: 'ML inference service timed out before returning a prediction.',
        },
        onPredict: noopPredict,
        onRefreshStatus: noopAsync,
        onReloadModel: noopAsync,
      })
    );

    assert.match(html, /HTTP 504/);
    assert.match(html, /ml_service_timeout/);
    assert.match(html, /ML inference service timed out/);
    assert.match(html, /FastAPI Offline/);
  });

  it('displays real FastAPI validation metrics in ModelMlflowView when artifact is loaded', () => {
    const html = renderToStaticMarkup(
      React.createElement(ModelMlflowView, {
        runs: [
          {
            runId: 'mlf-run-048',
            name: 'baseline',
            modelType: 'Blended Ensemble',
            createdAt: '2026-11-18',
            prAuc: 0.894,
            rocAuc: 0.986,
            f1AtThreshold: 0.842,
            precisionAt90Recall: 0.768,
            p99LatencyMs: 14.2,
            imbalanceStrategy: 'scale_pos_weight',
            blendWeightXgb: 0.65,
            blendWeightGnn: 0.35,
            status: 'production',
            dataset: 'IEEE-CIS',
          },
        ],
        prCurve: [{ recall: 0.5, xgbPrecision: 0.9, gnnPrecision: 0.8, blendedPrecision: 0.92 }],
        modelConfig: {
          activeRunId: 'mlf-run-048',
          xgbWeight: 0.65,
          gnnWeight: 0.35,
          reviewThreshold: 0.58,
          blockThreshold: 0.82,
        },
        mlServiceReachable: true,
        mlModelInfo: {
          ready: true,
          model_type: 'XGBoost Binary Classifier',
          model_version: 'xgb-real-v3',
          trained_at_utc: '2026-11-19T00:00:00Z',
          artifact_dir: 'artifacts/models/xgboost',
          missing_artifacts: [],
          feature_count: 54,
          feature_names: [],
          thresholds: { review: 0.44, block: 0.78 },
          validation_metrics: {
            metrics: { pr_auc: 0.7741, roc_auc: 0.9312, f1_score: 0.7155 },
          },
          pr_curve: null,
          gnn_status: 'not_implemented_phase4',
        },
        onUpdateModelConfig: noopAsync,
      })
    );

    assert.match(html, /xgb-real-v3/);
    assert.match(html, /0\.7741/);
    assert.match(html, /0\.9312/);
    assert.match(html, /0\.44 \/ 0\.78/);
  });
});
