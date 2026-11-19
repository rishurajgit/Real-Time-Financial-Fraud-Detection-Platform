import { useMemo, useState } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  ArrowUpRight,
  CheckCircle2,
  Search,
  ShieldAlert,
  SlidersHorizontal,
} from 'lucide-react';
import type {
  CaseStatus,
  DatasetProfile,
  DecisionAction,
  FeedbackLabel,
  ModelConfig,
  PlatformMetrics,
  ScoredTransaction,
  StreamControlState,
  ThroughputPoint,
} from '../types/fraud';

interface LiveStreamViewProps {
  metrics: PlatformMetrics;
  stream: StreamControlState;
  modelConfig: ModelConfig;
  transactions: ScoredTransaction[];
  throughputHistory: ThroughputPoint[];
  onUpdateStream: (patch: Partial<StreamControlState>) => Promise<void>;
  onUpdateCase: (
    caseId: string,
    patch: { status?: CaseStatus; feedbackLabel?: FeedbackLabel; notes?: string }
  ) => Promise<void>;
  onOpenRing: (ringId: string) => void;
}

const DECISION_FILTERS: Array<{ id: 'ALL' | 'FLAGGED' | DecisionAction; label: string }> = [
  { id: 'ALL', label: 'All Events' },
  { id: 'FLAGGED', label: 'Flagged Only' },
  { id: 'BLOCK', label: 'Blocked' },
  { id: 'REVIEW', label: 'Review' },
  { id: 'APPROVE', label: 'Approved' },
];

function formatUsd(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(amount);
}

function decisionLabel(decision: DecisionAction): { text: string; colorClass: string } {
  if (decision === 'BLOCK') return { text: 'Blocked', colorClass: 'text-red-700 font-semibold' };
  if (decision === 'REVIEW') return { text: 'Review', colorClass: 'text-amber-700 font-semibold' };
  return { text: 'Approved', colorClass: 'text-emerald-700 font-medium' };
}

export function LiveStreamView({
  metrics,
  stream,
  modelConfig,
  transactions,
  throughputHistory,
  onUpdateStream,
  onUpdateCase,
  onOpenRing,
}: LiveStreamViewProps) {
  const [filter, setFilter] = useState<'ALL' | 'FLAGGED' | DecisionAction>('ALL');
  const [query, setQuery] = useState('');
  const [selectedTxId, setSelectedTxId] = useState<string | null>(null);
  const [labelFeedbackNotice, setLabelFeedbackNotice] = useState<string | null>(null);

  const filteredTransactions = useMemo(() => {
    const q = query.trim().toLowerCase();
    return transactions.filter((tx) => {
      if (filter === 'FLAGGED' && tx.decision === 'APPROVE') return false;
      if (filter !== 'ALL' && filter !== 'FLAGGED' && tx.decision !== filter) return false;
      if (!q) return true;
      return (
        tx.id.toLowerCase().includes(q) ||
        tx.accountId.toLowerCase().includes(q) ||
        tx.merchantName.toLowerCase().includes(q) ||
        tx.location.toLowerCase().includes(q) ||
        (tx.ringId?.toLowerCase().includes(q) ?? false)
      );
    });
  }, [transactions, filter, query]);

  const selectedTx = useMemo(() => {
    if (selectedTxId) {
      const found = transactions.find((t) => t.id === selectedTxId);
      if (found) return found;
    }
    return filteredTransactions.find((t) => t.decision !== 'APPROVE') ?? filteredTransactions[0] ?? null;
  }, [selectedTxId, transactions, filteredTransactions]);

  const handleQuickLabel = async (tx: ScoredTransaction, label: 'fraud' | 'legitimate') => {
    if (!tx.caseId) return;
    await onUpdateCase(tx.caseId, {
      status: label === 'fraud' ? 'confirmed_fraud' : 'false_positive',
      feedbackLabel: label,
      notes:
        label === 'fraud'
          ? 'Confirmed fraudulent during live stream inspection.'
          : 'Cleared as legitimate cardholder activity from live stream.',
    });
    setLabelFeedbackNotice(
      label === 'fraud'
        ? `${tx.caseId} marked Confirmed Fraud and queued for Feast/MLflow feedback.`
        : `${tx.caseId} cleared as False Positive and recorded for threshold calibration.`
    );
    setTimeout(() => setLabelFeedbackNotice(null), 4000);
  };

  return (
    <div className="space-y-6">
      {/* KPI Overview Strip */}
      <section className="bg-white border border-slate-200 rounded-lg">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-slate-200">
          <div className="p-5">
            <p className="text-xs font-medium text-slate-500">Transactions Scored</p>
            <p className="mt-1.5 text-2xl font-semibold tracking-tight text-slate-900 font-mono tabular-nums">
              {metrics.totalProcessed.toLocaleString()}
            </p>
            <p className="mt-1 text-xs text-slate-500">
              <span>{stream.dataset} schema</span>
              <span className="mx-1.5" aria-hidden="true">·</span>
              <span className="font-mono tabular-nums">{metrics.flaggedCount} flagged</span>
            </p>
          </div>

          <div className="p-5">
            <p className="text-xs font-medium text-slate-500">Blocked Fraud Rate</p>
            <p className="mt-1.5 text-2xl font-semibold tracking-tight text-red-700 font-mono tabular-nums">
              {metrics.fraudRatePct.toFixed(2)}%
            </p>
            <p className="mt-1 text-xs text-slate-500">
              <span className="font-mono tabular-nums">{formatUsd(metrics.preventedLossUsd)}</span>
              <span className="mx-1.5" aria-hidden="true">·</span>
              <span>loss prevented</span>
            </p>
          </div>

          <div className="p-5">
            <p className="text-xs font-medium text-slate-500">Scoring Latency (Avg / p99)</p>
            <p className="mt-1.5 text-2xl font-semibold tracking-tight text-slate-900 font-mono tabular-nums">
              {metrics.avgLatencyMs.toFixed(1)} ms{' '}
              <span className="text-base font-normal text-slate-400">
                / {metrics.p99LatencyMs.toFixed(1)} ms
              </span>
            </p>
            <p className="mt-1 text-xs text-slate-500">
              <span>Feast lookup + XGBoost + PyG GNN</span>
            </p>
          </div>

          <div className="p-5">
            <p className="text-xs font-medium text-slate-500">Active Queue & Rings</p>
            <p className="mt-1.5 text-2xl font-semibold tracking-tight text-slate-900 font-mono tabular-nums">
              {metrics.openCases} cases{' '}
              <span className="text-base font-normal text-slate-400">
                / {metrics.activeRings} rings
              </span>
            </p>
            <p className="mt-1 text-xs text-slate-500">
              <span className="font-mono tabular-nums">{metrics.feedbackLabelsCollected}</span>
              <span className="ml-1">analyst labels stored</span>
            </p>
          </div>
        </div>
      </section>

      {/* Kafka Replay Throughput & Stream Controls */}
      <section className="bg-white border border-slate-200 rounded-lg p-5">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 pb-4 border-b border-slate-200">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">
              Kafka Consumer Throughput & Latency
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Active model {modelConfig.activeRunId} · Blend {Math.round(modelConfig.xgbWeight * 100)}% XGBoost /{' '}
              {Math.round(modelConfig.gnnWeight * 100)}% PyG GNN · Block threshold{' '}
              <span className="font-mono tabular-nums">{modelConfig.blockThreshold.toFixed(2)}</span>
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md">
              {(['IEEE-CIS', 'PaySim'] as DatasetProfile[]).map((ds) => (
                <button
                  key={ds}
                  type="button"
                  onClick={() => onUpdateStream({ dataset: ds })}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                    stream.dataset === ds
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {ds}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md">
              {[
                { label: '1x', rate: 0.9 },
                { label: '2x', rate: 1.5 },
                { label: '3x', rate: 2.4 },
              ].map((opt) => (
                <button
                  key={opt.label}
                  type="button"
                  onClick={() => onUpdateStream({ ratePerSec: opt.rate })}
                  className={`px-2.5 py-1 text-xs font-mono font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                    Math.abs(stream.ratePerSec - opt.rate) < 0.2
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="h-44 pt-4">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={throughputHistory} margin={{ top: 4, right: 12, left: -18, bottom: 0 }}>
              <defs>
                <linearGradient id="tpsFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#1D4ED8" stopOpacity={0.14} />
                  <stop offset="95%" stopColor="#1D4ED8" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 11, fill: '#64748B', fontFamily: 'JetBrains Mono' }}
                tickLine={false}
                axisLine={{ stroke: '#E2E8F0' }}
                minTickGap={28}
              />
              <YAxis
                tick={{ fontSize: 11, fill: '#64748B', fontFamily: 'JetBrains Mono' }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  borderColor: '#CBD5E1',
                  borderRadius: '6px',
                  fontSize: '12px',
                  fontFamily: 'JetBrains Mono',
                }}
              />
              <Area
                type="monotone"
                dataKey="tps"
                name="Events / sec"
                stroke="#1D4ED8"
                strokeWidth={1.75}
                fill="url(#tpsFill)"
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Main Workspace Split: Live Transaction Table + Feature & Score Inspector */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        {/* Live Transactions Table */}
        <section className="xl:col-span-8 bg-white border border-slate-200 rounded-lg overflow-hidden">
          <div className="p-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md overflow-x-auto">
              {DECISION_FILTERS.map((f) => (
                <button
                  key={f.id}
                  type="button"
                  onClick={() => setFilter(f.id)}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                    filter === f.id
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>

            <div className="relative w-full sm:w-60">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search TX, account, merchant..."
                className="w-full pl-8 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-md text-slate-900 placeholder:text-slate-400 focus:outline-none focus:bg-white focus:border-blue-600"
              />
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50/70 text-[11px] font-semibold text-slate-500">
                  <th className="py-2.5 px-4">Transaction / Account</th>
                  <th className="py-2.5 px-3">Merchant & Location</th>
                  <th className="py-2.5 px-3 text-right">Amount</th>
                  <th className="py-2.5 px-3 text-right">XGB</th>
                  <th className="py-2.5 px-3 text-right">GNN</th>
                  <th className="py-2.5 px-3 text-right">Risk Score</th>
                  <th className="py-2.5 px-4 text-right">Decision</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-xs">
                {filteredTransactions.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-12 text-center text-slate-500">
                      No transactions match the current filter criteria.
                    </td>
                  </tr>
                ) : (
                  filteredTransactions.slice(0, 22).map((tx) => {
                    const isSelected = selectedTx?.id === tx.id;
                    const statusMeta = decisionLabel(tx.decision);
                    return (
                      <tr
                        key={tx.id}
                        onClick={() => setSelectedTxId(tx.id)}
                        className={`cursor-pointer transition-colors ${
                          isSelected ? 'bg-blue-50/70' : 'hover:bg-slate-50'
                        }`}
                      >
                        <td className="py-2.5 px-4 whitespace-nowrap">
                          <div className="font-mono font-medium text-slate-900 tabular-nums">{tx.id}</div>
                          <div className="text-[11px] text-slate-500 font-mono tabular-nums">
                            {tx.accountId}
                            {tx.ringId ? ` · ${tx.ringId}` : ''}
                          </div>
                        </td>
                        <td className="py-2.5 px-3 max-w-[190px]">
                          <div className="font-medium text-slate-900 truncate">{tx.merchantName}</div>
                          <div className="text-[11px] text-slate-500 truncate">
                            {tx.location} · {tx.txType}
                          </div>
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums font-medium text-slate-900 whitespace-nowrap">
                          {formatUsd(tx.amount)}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-600">
                          {tx.xgbScore.toFixed(2)}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-600">
                          {tx.gnnScore.toFixed(2)}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums whitespace-nowrap">
                          <span
                            className={
                              tx.blendedScore >= modelConfig.blockThreshold
                                ? 'text-red-700 font-semibold'
                                : tx.blendedScore >= modelConfig.reviewThreshold
                                  ? 'text-amber-700 font-semibold'
                                  : 'text-slate-800'
                            }
                          >
                            {tx.blendedScore.toFixed(3)}
                          </span>
                        </td>
                        <td className="py-2.5 px-4 text-right whitespace-nowrap">
                          <span className={statusMeta.colorClass}>{statusMeta.text}</span>
                          <div className="text-[11px] font-mono tabular-nums text-slate-400">
                            {tx.latencyMs.toFixed(1)} ms
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </section>

        {/* Right Transaction & Feast Feature Inspector */}
        <aside className="xl:col-span-4 bg-white border border-slate-200 rounded-lg p-5">
          {!selectedTx ? (
            <div className="py-12 text-center text-xs text-slate-500">
              Select a transaction row to inspect Feast features and model attributions.
            </div>
          ) : (
            <div className="space-y-5">
              <div className="flex items-start justify-between gap-3 pb-4 border-b border-slate-200">
                <div>
                  <p className="text-xs text-slate-500 font-mono tabular-nums">
                    {selectedTx.id} · {selectedTx.timestamp.slice(11, 19)} UTC
                  </p>
                  <h3 className="text-base font-semibold text-slate-900 mt-0.5">
                    {selectedTx.merchantName}
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    {selectedTx.accountId} · {selectedTx.location}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-lg font-semibold font-mono tabular-nums text-slate-900">
                    {formatUsd(selectedTx.amount)}
                  </p>
                  <p className={`text-xs ${decisionLabel(selectedTx.decision).colorClass}`}>
                    {decisionLabel(selectedTx.decision).text} ({selectedTx.blendedScore.toFixed(3)})
                  </p>
                </div>
              </div>

              {/* Blended Model Score Breakdown */}
              <div className="space-y-2.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700 flex items-center gap-1.5">
                    <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400" />
                    Ensemble Breakdown
                  </span>
                  <span className="font-mono tabular-nums text-slate-500">
                    {selectedTx.latencyMs.toFixed(1)} ms total
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2 pt-1 text-xs">
                  <div className="p-2.5 bg-slate-50 rounded-md">
                    <p className="text-[11px] text-slate-500">XGBoost</p>
                    <p className="mt-0.5 font-mono tabular-nums font-semibold text-slate-900">
                      {selectedTx.xgbScore.toFixed(3)}
                    </p>
                  </div>
                  <div className="p-2.5 bg-slate-50 rounded-md">
                    <p className="text-[11px] text-slate-500">PyG GNN</p>
                    <p className="mt-0.5 font-mono tabular-nums font-semibold text-slate-900">
                      {selectedTx.gnnScore.toFixed(3)}
                    </p>
                  </div>
                  <div className="p-2.5 bg-slate-50 rounded-md">
                    <p className="text-[11px] text-slate-500">Feast Lookup</p>
                    <p className="mt-0.5 font-mono tabular-nums font-semibold text-slate-900">
                      {selectedTx.features.feastLookupMs.toFixed(1)} ms
                    </p>
                  </div>
                </div>
              </div>

              {/* SHAP-style Feature Attribution Bars */}
              <div className="pt-4 border-t border-slate-200 space-y-2.5">
                <h4 className="text-xs font-semibold text-slate-900">
                  Top Risk Drivers (Log-Odds Contribution)
                </h4>
                <div className="space-y-2">
                  {selectedTx.attributions.slice(0, 5).map((attr) => {
                    const isRisk = attr.contribution >= 0;
                    const widthPct = Math.min(100, Math.round((Math.abs(attr.contribution) / 2.4) * 100));
                    return (
                      <div key={attr.feature} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-slate-700 truncate pr-2">{attr.label}</span>
                          <span className="font-mono tabular-nums text-slate-600 whitespace-nowrap shrink-0">
                            {attr.value}{' '}
                            <span className={isRisk ? 'text-red-700 font-medium' : 'text-emerald-700'}>
                              ({isRisk ? '+' : ''}
                              {attr.contribution.toFixed(2)})
                            </span>
                          </span>
                        </div>
                        <div className="h-1.5 bg-slate-100 rounded-xs overflow-hidden">
                          <div
                            className={`h-full ${isRisk ? 'bg-red-600' : 'bg-emerald-600'}`}
                            style={{ width: `${Math.max(4, widthPct)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Feast Online Feature Vector */}
              <div className="pt-4 border-t border-slate-200 space-y-2">
                <h4 className="text-xs font-semibold text-slate-900">Feast Online Feature Vector</h4>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                  <div className="flex justify-between">
                    <dt className="text-slate-500">tx_count_1h</dt>
                    <dd className="font-mono tabular-nums text-slate-900">{selectedTx.features.txCount1h}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-500">tx_count_24h</dt>
                    <dd className="font-mono tabular-nums text-slate-900">{selectedTx.features.txCount24h}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-500">amt_vs_avg</dt>
                    <dd className="font-mono tabular-nums text-slate-900">
                      {selectedTx.features.amountVsUserAvg.toFixed(2)}x
                    </dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-500">device_change</dt>
                    <dd className="font-mono tabular-nums text-slate-900">
                      {selectedTx.features.deviceChangeFlag}
                    </dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-500">geo_delta_km</dt>
                    <dd className="font-mono tabular-nums text-slate-900">
                      {selectedTx.features.geoDistanceKm}
                    </dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-500">shared_accts</dt>
                    <dd className="font-mono tabular-nums text-slate-900">
                      {selectedTx.features.cardSharedAccounts}
                    </dd>
                  </div>
                </dl>
                <div className="pt-1 text-[11px] text-slate-500 font-mono truncate">
                  Device: {selectedTx.deviceId} · Card: {selectedTx.cardHash}
                </div>
              </div>

              {/* Analyst Feedback & Ring Navigation Actions */}
              <div className="pt-4 border-t border-slate-200 space-y-2.5">
                {selectedTx.ringId && (
                  <button
                    type="button"
                    onClick={() => onOpenRing(selectedTx.ringId!)}
                    className="w-full flex items-center justify-between px-3.5 py-2 text-xs font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 rounded-md transition-colors whitespace-nowrap"
                  >
                    <span>Inspect Linked Fraud Ring ({selectedTx.ringId})</span>
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  </button>
                )}

                {selectedTx.caseId ? (
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => handleQuickLabel(selectedTx, 'fraud')}
                      className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-white bg-red-600 hover:bg-red-700 rounded-md transition-colors whitespace-nowrap"
                    >
                      <ShieldAlert className="w-3.5 h-3.5" />
                      Confirm Fraud
                    </button>
                    <button
                      type="button"
                      onClick={() => handleQuickLabel(selectedTx, 'legitimate')}
                      className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-md transition-colors whitespace-nowrap"
                    >
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      Mark Legitimate
                    </button>
                  </div>
                ) : (
                  <p className="text-xs text-slate-500">
                    Cleared automatically below review threshold ({modelConfig.reviewThreshold.toFixed(2)}).
                  </p>
                )}

                {labelFeedbackNotice && (
                  <p className="text-xs text-emerald-700 bg-emerald-50 px-3 py-2 rounded-md">
                    {labelFeedbackNotice}
                  </p>
                )}
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
