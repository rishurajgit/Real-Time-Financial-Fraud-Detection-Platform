import { useState } from 'react';
import { ArrowRight, Play } from 'lucide-react';
import type {
  DatasetEdaSummary,
  DatasetProfile,
  ScoreTransactionRequest,
  ScoredTransaction,
} from '../types/fraud';
import type { MlInferenceActions } from './LiveStreamView';
import { RealMlInferencePanel } from './RealMlInferencePanel';

interface ArchitectureApiViewProps {
  edaSummaries: Record<DatasetProfile, DatasetEdaSummary>;
  activeDataset: DatasetProfile;
  mlInference?: MlInferenceActions;
  onScoreTransaction: (req: ScoreTransactionRequest) => Promise<ScoredTransaction>;
}

const PRESETS: Array<{ label: string; payload: ScoreTransactionRequest }> = [
  {
    label: 'Fraud Ring Attack',
    payload: {
      accountId: 'ACC-88101',
      amount: 2480.0,
      merchantName: 'BitVault Exchange',
      merchantCategory: 'Crypto On-Ramp',
      location: 'Bucharest, RO',
      deviceId: 'DEV-F991A',
      cardHash: 'CARD-****-8841',
      ipAddress: '185.220.101.44',
      txType: 'CARD_NOT_PRESENT',
      txCount1h: 7,
      geoDistanceKm: 4820,
      deviceChangeFlag: 1,
      ingestIntoStream: true,
    },
  },
  {
    label: 'Cross-Border Takeover',
    payload: {
      accountId: 'ACC-10051',
      amount: 1120.5,
      merchantName: 'Nordic Luxury Watches',
      merchantCategory: 'Luxury Retail',
      location: 'Lagos, NG',
      deviceId: 'DEV-NEW-772',
      cardHash: 'CARD-****-2229',
      ipAddress: '197.210.84.12',
      txType: 'DIGITAL_WALLET',
      txCount1h: 4,
      geoDistanceKm: 5340,
      deviceChangeFlag: 1,
      ingestIntoStream: true,
    },
  },
  {
    label: 'Legitimate Grocery',
    payload: {
      accountId: 'ACC-10017',
      amount: 64.2,
      merchantName: 'Whole Foods Market',
      merchantCategory: 'Grocery',
      location: 'New York, US',
      deviceId: 'DEV-407',
      cardHash: 'CARD-****-2143',
      ipAddress: '104.28.42.19',
      txType: 'POS_CHIP',
      txCount1h: 1,
      geoDistanceKm: 4,
      deviceChangeFlag: 0,
      ingestIntoStream: true,
    },
  },
];

const ARCHITECTURE_STAGES = [
  {
    step: '01',
    title: 'Kafka Replay Producer',
    stack: 'Apache Kafka · Avro Schema',
    detail: 'Replays IEEE-CIS & PaySim events onto transactions.live topic partitioned by account_id.',
  },
  {
    step: '02',
    title: 'Feast Online Store',
    stack: 'Feast · Redis + Parquet',
    detail: 'Serves identical 1h/24h velocity, amount ratio, and device-shift vectors in under 4 ms.',
  },
  {
    step: '03',
    title: 'FastAPI Scoring Service',
    stack: 'FastAPI · XGBoost + PyG',
    detail: 'Blends class-weighted XGBoost log-odds with 2-hop GraphSAGE ring embeddings.',
  },
  {
    step: '04',
    title: 'MLflow & Evidently AI',
    stack: 'MLflow · Evidently · K8s',
    detail: 'Tracks PR-AUC runs, monitors PSI drift continuously, and triggers Kubernetes retraining jobs.',
  },
];

export function ArchitectureApiView({
  edaSummaries,
  activeDataset,
  mlInference,
  onScoreTransaction,
}: ArchitectureApiViewProps) {
  const [edaProfile, setEdaProfile] = useState<DatasetProfile>(activeDataset);
  const [form, setForm] = useState<ScoreTransactionRequest>(PRESETS[0].payload);
  const [response, setResponse] = useState<ScoredTransaction | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const activeEda = edaSummaries[edaProfile];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setErrorMsg(null);
    try {
      const result = await onScoreTransaction(form);
      setResponse(result);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Request failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Architecture Pipeline Blueprint */}
      <section className="bg-white border border-slate-200 rounded-lg p-5">
        <div className="pb-4 border-b border-slate-200">
          <h2 className="text-sm font-semibold text-slate-900">
            End-to-End Platform Architecture (Kubernetes Deployment)
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Training-serving feature parity via Feast with sub-20 ms FastAPI inference pods autoscaled on Minikube / EKS.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 pt-4">
          {ARCHITECTURE_STAGES.map((stage, idx) => (
            <div
              key={stage.step}
              className="p-4 bg-slate-50/80 border border-slate-200 rounded-md flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono font-semibold text-blue-700">{stage.step}</span>
                  {idx < ARCHITECTURE_STAGES.length - 1 && (
                    <ArrowRight className="w-3.5 h-3.5 text-slate-400 hidden xl:block" />
                  )}
                </div>
                <h3 className="mt-2 text-sm font-semibold text-slate-900">{stage.title}</h3>
                <p className="mt-0.5 text-[11px] font-mono text-slate-500">{stage.stack}</p>
                <p className="mt-2 text-xs text-slate-600 leading-relaxed">{stage.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Dataset EDA & Class Imbalance Section */}
      <section className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">
              Dataset EDA & Severe Class Imbalance Profile
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Under 1% positive fraud labels — why standard accuracy fails and PR-AUC with class weighting is mandatory.
            </p>
          </div>
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md self-start">
            {(['IEEE-CIS', 'PaySim'] as DatasetProfile[]).map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => setEdaProfile(p)}
                className={`px-3 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap ${
                  edaProfile === p ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div className="p-3.5 bg-slate-50 rounded-md">
            <p className="text-slate-500">Total Transactions</p>
            <p className="mt-1 text-lg font-semibold font-mono tabular-nums text-slate-900">
              {activeEda.totalRows}
            </p>
          </div>
          <div className="p-3.5 bg-slate-50 rounded-md">
            <p className="text-slate-500">Fraud Incidence</p>
            <p className="mt-1 text-lg font-semibold font-mono tabular-nums text-red-700">
              {activeEda.fraudRatioPct.toFixed(2)}%
            </p>
          </div>
          <div className="p-3.5 bg-slate-50 rounded-md">
            <p className="text-slate-500">Imbalance Ratio</p>
            <p className="mt-1 text-lg font-semibold font-mono tabular-nums text-slate-900">
              {activeEda.imbalanceRatio}
            </p>
          </div>
          <div className="p-3.5 bg-slate-50 rounded-md">
            <p className="text-slate-500">Feast Feature Views</p>
            <p className="mt-1 text-lg font-semibold font-mono tabular-nums text-slate-900">
              {activeEda.feastFeatureCount} features
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-[11px] font-semibold text-slate-500">
                <th className="py-2 pr-4">Engineered Feast Feature</th>
                <th className="py-2 px-3 text-right">Fraud Median</th>
                <th className="py-2 px-3 text-right">Legitimate Median</th>
                <th className="py-2 pl-3 text-right">Signal Lift</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {activeEda.keySignals.map((sig) => (
                <tr key={sig.feature}>
                  <td className="py-2.5 pr-4 font-mono text-slate-900">{sig.feature}</td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums text-red-700 font-medium">
                    {sig.fraudMedian}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-600">
                    {sig.legitMedian}
                  </td>
                  <td className="py-2.5 pl-3 text-right font-mono tabular-nums font-semibold text-blue-700">
                    {sig.lift}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Real FastAPI XGBoost Prediction Sandbox */}
      {mlInference && (
        <RealMlInferencePanel
          health={mlInference.health}
          modelInfo={mlInference.modelInfo}
          serviceReachable={mlInference.serviceReachable}
          statusLoading={mlInference.statusLoading}
          predicting={mlInference.predicting}
          reloading={mlInference.reloading}
          lastPrediction={mlInference.lastPrediction}
          realPredictions={mlInference.realPredictions}
          error={mlInference.error}
          onPredict={mlInference.predictTransaction}
          onRefreshStatus={mlInference.refreshStatus}
          onReloadModel={mlInference.reloadModel}
        />
      )}

      {/* Legacy Simulated Express Scoring Endpoint Sandbox */}
      <section className="bg-white border border-slate-200 rounded-lg p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">
              Legacy Simulated Scoring Sandbox — POST /api/v1/score
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              In-memory Express simulator preserved for historical replay testing (does not invoke XGBoost).
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                onClick={() => setForm(preset.payload)}
                className="px-2.5 py-1 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded transition-colors whitespace-nowrap"
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 pt-4">
          <form onSubmit={handleSubmit} className="xl:col-span-6 space-y-3 text-xs">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-slate-600 mb-1">Account ID</label>
                <input
                  type="text"
                  value={form.accountId}
                  onChange={(e) => setForm({ ...form, accountId: e.target.value })}
                  className="w-full px-3 py-1.5 font-mono bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:bg-white focus:border-blue-600"
                />
              </div>
              <div>
                <label className="block text-slate-600 mb-1">Amount (USD)</label>
                <input
                  type="number"
                  step="0.01"
                  value={form.amount}
                  onChange={(e) => setForm({ ...form, amount: Number(e.target.value) })}
                  className="w-full px-3 py-1.5 font-mono bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:bg-white focus:border-blue-600"
                />
              </div>
              <div>
                <label className="block text-slate-600 mb-1">Merchant</label>
                <input
                  type="text"
                  value={form.merchantName}
                  onChange={(e) => setForm({ ...form, merchantName: e.target.value })}
                  className="w-full px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:bg-white focus:border-blue-600"
                />
              </div>
              <div>
                <label className="block text-slate-600 mb-1">Location</label>
                <input
                  type="text"
                  value={form.location}
                  onChange={(e) => setForm({ ...form, location: e.target.value })}
                  className="w-full px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:bg-white focus:border-blue-600"
                />
              </div>
              <div>
                <label className="block text-slate-600 mb-1">Device Fingerprint</label>
                <input
                  type="text"
                  value={form.deviceId}
                  onChange={(e) => setForm({ ...form, deviceId: e.target.value })}
                  className="w-full px-3 py-1.5 font-mono bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:bg-white focus:border-blue-600"
                />
              </div>
              <div>
                <label className="block text-slate-600 mb-1">1h Velocity (tx/hr)</label>
                <input
                  type="number"
                  min={1}
                  max={25}
                  value={form.txCount1h ?? 1}
                  onChange={(e) => setForm({ ...form, txCount1h: Number(e.target.value) })}
                  className="w-full px-3 py-1.5 font-mono bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:bg-white focus:border-blue-600"
                />
              </div>
            </div>

            <div className="flex items-center justify-between pt-2">
              <label className="flex items-center gap-2 text-slate-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={form.ingestIntoStream !== false}
                  onChange={(e) => setForm({ ...form, ingestIntoStream: e.target.checked })}
                  className="accent-blue-700"
                />
                <span>Ingest scored event into live Kafka stream & case queue</span>
              </label>
              <button
                type="submit"
                disabled={submitting}
                className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-white bg-blue-700 hover:bg-blue-800 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
              >
                <Play className="w-3.5 h-3.5" />
                {submitting ? 'Scoring...' : 'Send Score Request'}
              </button>
            </div>

            {errorMsg && <p className="text-xs text-red-700">{errorMsg}</p>}
          </form>

          <div className="xl:col-span-6 bg-slate-900 text-slate-100 rounded-md p-4 font-mono text-[11px] overflow-x-auto max-h-72 overflow-y-auto">
            {response ? (
              <pre className="leading-relaxed">
                {JSON.stringify(
                  {
                    transaction_id: response.id,
                    decision: response.decision,
                    latency_ms: response.latencyMs,
                    scores: {
                      blended_risk_score: response.blendedScore,
                      xgboost_score: response.xgbScore,
                      pyg_gnn_score: response.gnnScore,
                    },
                    fraud_ring_id: response.ringId,
                    case_created: response.caseId,
                    feast_features: response.features,
                  },
                  null,
                  2
                )}
              </pre>
            ) : (
              <div className="h-44 flex items-center justify-center text-slate-400">
                Submit a transaction payload to inspect live JSON response from /api/v1/score.
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
