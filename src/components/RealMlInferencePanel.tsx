import { useState } from 'react';
import { AlertTriangle, CheckCircle2, Loader2, Play, RefreshCw } from 'lucide-react';
import type {
  MlApiError,
  MlDecision,
  MlHealthStatus,
  MlModelInfo,
  MlPredictRequest,
  MlPredictionResponse,
} from '../types/fraud';

export interface RealMlInferencePanelProps {
  health: MlHealthStatus | null;
  modelInfo: MlModelInfo | null;
  serviceReachable: boolean;
  statusLoading: boolean;
  predicting: boolean;
  reloading: boolean;
  lastPrediction: MlPredictionResponse | null;
  realPredictions: MlPredictionResponse[];
  error: MlApiError | null;
  onPredict: (payload: MlPredictRequest) => Promise<MlPredictionResponse | null>;
  onRefreshStatus: () => Promise<void>;
  onReloadModel: () => Promise<void>;
}

const DEFAULT_PAYLOAD: MlPredictRequest = {
  transaction_id: 'IEEE-LIVE-1001',
  transaction_dt: 86400,
  transaction_amt: 645.5,
  product_cd: 'C',
  card1: 13926,
  card4: 'visa',
  card6: 'credit',
  addr1: 315,
  addr2: 87,
  dist1: 19,
  p_emaildomain: 'gmail.com',
  device_type: 'mobile',
  device_info: 'SM-G991B',
};

function decisionStyle(decision: MlDecision): string {
  if (decision === 'Blocked') return 'text-red-700 font-semibold';
  if (decision === 'Review') return 'text-amber-700 font-semibold';
  return 'text-emerald-700 font-semibold';
}

export function RealMlInferencePanel({
  health,
  modelInfo,
  serviceReachable,
  statusLoading,
  predicting,
  reloading,
  lastPrediction,
  realPredictions,
  error,
  onPredict,
  onRefreshStatus,
  onReloadModel,
}: RealMlInferencePanelProps) {
  const [form, setForm] = useState<MlPredictRequest>(DEFAULT_PAYLOAD);

  const modelReady = Boolean(serviceReachable && health?.model_ready);
  const missingArtifacts =
    modelInfo?.missing_artifacts?.length
      ? modelInfo.missing_artifacts
      : health?.missing_artifacts?.length
        ? health.missing_artifacts
        : error?.missing_artifacts ?? [];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onPredict(form);
  };

  return (
    <section
      className="bg-white border border-slate-200 rounded-lg p-5 space-y-5"
      aria-label="FastAPI XGBoost Inference"
    >
      {/* Header & Real Model Readiness */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 pb-4 border-b border-slate-200">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">
            Real XGBoost Inference (FastAPI via /api/v1/ml/predict)
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            {statusLoading ? (
              'Checking FastAPI service and XGBoost model readiness...'
            ) : modelReady ? (
              <>
                <span className="text-emerald-700 font-semibold">Model Ready</span>
                <span className="mx-1.5" aria-hidden="true">·</span>
                <span className="font-mono text-slate-700">{health?.model_version}</span>
                {modelInfo?.feature_count ? (
                  <>
                    <span className="mx-1.5" aria-hidden="true">·</span>
                    <span className="font-mono tabular-nums">{modelInfo.feature_count} features</span>
                  </>
                ) : null}
                <span className="mx-1.5" aria-hidden="true">·</span>
                <span>XGBoost only (GNN scheduled Phase 4)</span>
              </>
            ) : serviceReachable ? (
              <>
                <span className="text-amber-700 font-semibold">Model Unavailable (HTTP 503)</span>
                <span className="mx-1.5" aria-hidden="true">·</span>
                <span>FastAPI online, trained XGBoost artifact not loaded</span>
              </>
            ) : (
              <>
                <span className="text-red-700 font-semibold">FastAPI Offline</span>
                <span className="mx-1.5" aria-hidden="true">·</span>
                <span>Start FastAPI on port 8000 to execute live XGBoost predictions</span>
              </>
            )}
          </p>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            disabled={statusLoading}
            onClick={onRefreshStatus}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 border border-slate-200 hover:bg-slate-50 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${statusLoading ? 'animate-spin' : ''}`} />
            Check Readiness
          </button>
          <button
            type="button"
            disabled={reloading || !serviceReachable}
            onClick={onReloadModel}
            className="px-3 py-1.5 text-xs font-medium text-slate-700 border border-slate-200 hover:bg-slate-50 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
          >
            {reloading ? 'Reloading...' : 'Reload Artifact'}
          </button>
        </div>
      </div>

      {/* Model Unavailable / Error Banner */}
      {!statusLoading && !modelReady && (
        <div
          role="status"
          className="bg-amber-50 border border-amber-200 rounded-md px-4 py-3 text-xs text-amber-900 space-y-1"
        >
          <div className="flex items-center gap-2 font-semibold">
            <AlertTriangle className="w-4 h-4 text-amber-700 shrink-0" />
            <span>
              {serviceReachable
                ? 'Model Unavailable — Trained Phase 1 XGBoost Artifacts Missing'
                : 'ML Inference Service Unreachable'}
            </span>
          </div>
          <p className="text-amber-800">
            {serviceReachable
              ? 'Predictions are disabled (HTTP 503) rather than returning simulated scores. Train Phase 1 with IEEE-CIS CSVs and click Reload Artifact.'
              : 'No random fallback predictions will be substituted. Start uvicorn on port 8000 and verify ML_SERVICE_URL.'}
          </p>
          {missingArtifacts.length > 0 && (
            <p className="font-mono text-[11px] text-amber-900">
              Missing: {missingArtifacts.join(', ')}
            </p>
          )}
        </div>
      )}

      {error && (
        <div
          role="alert"
          className="bg-red-50 border border-red-200 rounded-md px-4 py-3 text-xs text-red-900 space-y-1"
        >
          <p className="font-semibold">
            Inference Error (HTTP {error.status} · {error.error})
          </p>
          <p className="text-red-800">{error.detail}</p>
        </div>
      )}

      {/* Prediction Form + Real Output Panel */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        <form onSubmit={handleSubmit} className="xl:col-span-6 space-y-3 text-xs">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div>
              <label htmlFor="ml-tx-id" className="block text-slate-600 mb-1">
                Transaction ID
              </label>
              <input
                id="ml-tx-id"
                type="text"
                required
                value={form.transaction_id}
                onChange={(e) => setForm({ ...form, transaction_id: e.target.value })}
                className="w-full px-2.5 py-1.5 font-mono bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:bg-white focus:border-blue-600"
              />
            </div>
            <div>
              <label htmlFor="ml-tx-amt" className="block text-slate-600 mb-1">
                Amount (USD)
              </label>
              <input
                id="ml-tx-amt"
                type="number"
                min={0}
                step="0.01"
                required
                value={form.transaction_amt}
                onChange={(e) => setForm({ ...form, transaction_amt: Number(e.target.value) })}
                className="w-full px-2.5 py-1.5 font-mono bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:bg-white focus:border-blue-600"
              />
            </div>
            <div>
              <label htmlFor="ml-tx-dt" className="block text-slate-600 mb-1">
                TransactionDT (sec)
              </label>
              <input
                id="ml-tx-dt"
                type="number"
                min={0}
                required
                value={form.transaction_dt}
                onChange={(e) => setForm({ ...form, transaction_dt: Number(e.target.value) })}
                className="w-full px-2.5 py-1.5 font-mono bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:bg-white focus:border-blue-600"
              />
            </div>
            <div>
              <label htmlFor="ml-card1" className="block text-slate-600 mb-1">
                card1
              </label>
              <input
                id="ml-card1"
                type="number"
                value={form.card1 ?? ''}
                onChange={(e) =>
                  setForm({ ...form, card1: e.target.value ? Number(e.target.value) : undefined })
                }
                className="w-full px-2.5 py-1.5 font-mono bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:bg-white focus:border-blue-600"
              />
            </div>
            <div>
              <label htmlFor="ml-product" className="block text-slate-600 mb-1">
                ProductCD
              </label>
              <select
                id="ml-product"
                value={form.product_cd ?? 'W'}
                onChange={(e) => setForm({ ...form, product_cd: e.target.value })}
                className="w-full px-2.5 py-1.5 bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:bg-white focus:border-blue-600"
              >
                <option value="W">W</option>
                <option value="C">C</option>
                <option value="H">H</option>
                <option value="R">R</option>
                <option value="S">S</option>
              </select>
            </div>
            <div>
              <label htmlFor="ml-device" className="block text-slate-600 mb-1">
                DeviceInfo
              </label>
              <input
                id="ml-device"
                type="text"
                value={form.device_info ?? ''}
                onChange={(e) => setForm({ ...form, device_info: e.target.value || undefined })}
                className="w-full px-2.5 py-1.5 font-mono bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:bg-white focus:border-blue-600"
              />
            </div>
          </div>

          <div className="flex items-center justify-between pt-1 gap-3">
            <span className="text-[11px] text-slate-500">
              Submits directly to Express <span className="font-mono">POST /api/v1/ml/predict</span>
            </span>
            <button
              type="submit"
              disabled={predicting}
              className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-white bg-blue-700 hover:bg-blue-800 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap shrink-0"
            >
              {predicting ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Scoring via XGBoost...
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" />
                  Run Real XGBoost Prediction
                </>
              )}
            </button>
          </div>
        </form>

        {/* Latest Real Prediction Output */}
        <div className="xl:col-span-6 bg-slate-50 border border-slate-200 rounded-md p-4 text-xs">
          {predicting ? (
            <div className="py-8 flex items-center justify-center gap-2 text-slate-600">
              <Loader2 className="w-4 h-4 animate-spin text-blue-700" />
              <span>Running real XGBoost predict_proba() inference...</span>
            </div>
          ) : lastPrediction ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between pb-2.5 border-b border-slate-200">
                <div>
                  <span className="font-mono font-semibold text-slate-900">
                    {lastPrediction.transaction_id}
                  </span>
                  <span className="mx-1.5 text-slate-400" aria-hidden="true">·</span>
                  <span className="text-emerald-700 font-medium inline-flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    Real XGBoost Inference
                  </span>
                </div>
                <span className={decisionStyle(lastPrediction.decision)}>
                  {lastPrediction.decision}
                </span>
              </div>

              <dl className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div>
                  <dt className="text-[11px] text-slate-500">Fraud Probability</dt>
                  <dd className="mt-0.5 font-mono tabular-nums text-sm font-semibold text-slate-900">
                    {(lastPrediction.xgboost_probability * 100).toFixed(2)}%
                  </dd>
                </div>
                <div>
                  <dt className="text-[11px] text-slate-500">Risk Score</dt>
                  <dd className="mt-0.5 font-mono tabular-nums text-sm font-semibold text-slate-900">
                    {lastPrediction.final_score.toFixed(4)}
                  </dd>
                </div>
                <div>
                  <dt className="text-[11px] text-slate-500">Latency</dt>
                  <dd className="mt-0.5 font-mono tabular-nums text-sm font-semibold text-slate-900">
                    {lastPrediction.inference_latency_ms.toFixed(2)} ms
                  </dd>
                </div>
                <div>
                  <dt className="text-[11px] text-slate-500">Model Version</dt>
                  <dd className="mt-0.5 font-mono text-xs font-semibold text-slate-900 truncate">
                    {lastPrediction.model_version}
                  </dd>
                </div>
              </dl>

              {lastPrediction.risk_indicators.length > 0 && (
                <div className="pt-2 border-t border-slate-200 space-y-1">
                  <p className="text-[11px] font-semibold text-slate-600">Observed Feature Signals</p>
                  <ul className="space-y-1 text-[11px] text-slate-700">
                    {lastPrediction.risk_indicators.slice(0, 4).map((ind) => (
                      <li key={ind.feature} className="flex items-center justify-between gap-2">
                        <span className="font-mono text-slate-600">{ind.feature}</span>
                        <span className="truncate">{ind.signal}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <div className="py-8 text-center text-slate-500">
              No real XGBoost predictions submitted yet. Run a prediction above once FastAPI reports Model Ready.
            </div>
          )}
        </div>
      </div>

      {/* Real Model-Scored Transactions History (strictly separated from simulated replay) */}
      {realPredictions.length > 0 && (
        <div className="pt-4 border-t border-slate-200 space-y-2">
          <h3 className="text-xs font-semibold text-slate-900">
            Real Model-Scored Transactions ({realPredictions.length})
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-[11px] font-semibold text-slate-500">
                  <th className="py-2 pr-3">Transaction ID</th>
                  <th className="py-2 px-3 text-right">XGBoost Prob</th>
                  <th className="py-2 px-3 text-right">Risk Score</th>
                  <th className="py-2 px-3">Model Version</th>
                  <th className="py-2 px-3 text-right">Latency</th>
                  <th className="py-2 pl-3 text-right">Decision</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {realPredictions.slice(0, 8).map((pred) => (
                  <tr key={`${pred.transaction_id}-${pred.scored_at_utc}`}>
                    <td className="py-2 pr-3 font-mono font-medium text-slate-900">
                      {pred.transaction_id}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-700">
                      {(pred.xgboost_probability * 100).toFixed(2)}%
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums font-semibold text-slate-900">
                      {pred.final_score.toFixed(4)}
                    </td>
                    <td className="py-2 px-3 font-mono text-slate-600">{pred.model_version}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-600">
                      {pred.inference_latency_ms.toFixed(2)} ms
                    </td>
                    <td className={`py-2 pl-3 text-right ${decisionStyle(pred.decision)}`}>
                      {pred.decision}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}
