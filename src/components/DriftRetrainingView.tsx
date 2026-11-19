import { useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Activity, AlertTriangle, RefreshCw, Zap } from 'lucide-react';
import type { DriftReport, StreamControlState } from '../types/fraud';

interface DriftRetrainingViewProps {
  driftReport: DriftReport;
  stream: StreamControlState;
  onToggleDriftInjection: (enable: boolean) => Promise<void>;
  onUpdateDriftConfig: (patch: { autoRetrainEnabled?: boolean; driftThresholdPsi?: number }) => Promise<void>;
  onTriggerRetrain: (reason?: string) => Promise<void>;
}

export function DriftRetrainingView({
  driftReport,
  stream,
  onToggleDriftInjection,
  onUpdateDriftConfig,
  onTriggerRetrain,
}: DriftRetrainingViewProps) {
  const [selectedFeatureKey, setSelectedFeatureKey] = useState<string>(
    driftReport.features[0]?.feature ?? 'amount_vs_user_avg'
  );
  const [busy, setBusy] = useState(false);

  const selectedFeature =
    driftReport.features.find((f) => f.feature === selectedFeatureKey) ?? driftReport.features[0];

  const isRunningJob = driftReport.jobs.some((j) => j.status === 'running');

  const handleManualRetrain = async () => {
    setBusy(true);
    try {
      await onTriggerRetrain('Manual analyst retraining triggered from Evidently AI monitor');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Drift Status KPI Strip */}
      <section className="bg-white border border-slate-200 rounded-lg">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-slate-200">
          <div className="p-5">
            <p className="text-xs text-slate-500">Dataset Drift (Mean PSI)</p>
            <p
              className={`mt-1.5 text-2xl font-semibold font-mono tabular-nums ${
                driftReport.driftAlertActive ? 'text-red-700' : 'text-emerald-700'
              }`}
            >
              {driftReport.datasetPsi.toFixed(3)}
            </p>
            <p className="mt-1 text-xs text-slate-500">
              Threshold: {driftReport.driftThresholdPsi.toFixed(2)} · Window {driftReport.windowSize} tx
            </p>
          </div>

          <div className="p-5">
            <p className="text-xs text-slate-500">Drifted Features</p>
            <p className="mt-1.5 text-2xl font-semibold font-mono tabular-nums text-slate-900">
              {driftReport.driftedFeatureCount}{' '}
              <span className="text-base font-normal text-slate-400">/ {driftReport.totalFeatures}</span>
            </p>
            <p className="mt-1 text-xs text-slate-500">
              {driftReport.driftAlertActive ? 'Drift alert active' : 'Within reference bounds'}
            </p>
          </div>

          <div className="p-5">
            <p className="text-xs text-slate-500">Auto-Retrain Trigger</p>
            <div className="mt-1.5 flex items-center justify-between gap-2">
              <span className="text-lg font-semibold text-slate-900">
                {driftReport.autoRetrainEnabled ? 'Armed' : 'Paused'}
              </span>
              <button
                type="button"
                onClick={() => onUpdateDriftConfig({ autoRetrainEnabled: !driftReport.autoRetrainEnabled })}
                className="px-2.5 py-1 text-xs font-medium text-slate-700 border border-slate-200 hover:bg-slate-50 rounded transition-colors whitespace-nowrap"
              >
                {driftReport.autoRetrainEnabled ? 'Pause Auto' : 'Enable Auto'}
              </button>
            </div>
            <p className="mt-1 text-xs text-slate-500">Fires Kubernetes retrain job on breach</p>
          </div>

          <div className="p-5 flex flex-col justify-between gap-2">
            <p className="text-xs text-slate-500">Drift Stress Simulation</p>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => onToggleDriftInjection(!stream.driftInjected)}
                className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                  stream.driftInjected
                    ? 'bg-amber-600 hover:bg-amber-700 text-white'
                    : 'bg-slate-900 hover:bg-slate-800 text-white'
                }`}
              >
                <Zap className="w-3.5 h-3.5" />
                {stream.driftInjected ? 'Stop Drift Shift' : 'Simulate Drift Shift'}
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Active Alert Notice */}
      {driftReport.driftAlertActive && (
        <div className="bg-red-50 border border-red-200 rounded-lg px-5 py-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-start gap-2.5 text-xs text-red-900">
            <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">
                Evidently AI Drift Threshold Breached (PSI {driftReport.datasetPsi.toFixed(3)} &ge;{' '}
                {driftReport.driftThresholdPsi.toFixed(2)})
              </p>
              <p className="text-red-700 mt-0.5">
                Live feature distributions have diverged from the training reference window.{' '}
                {driftReport.autoRetrainEnabled
                  ? 'Automatic retraining pipeline has been armed and scheduled.'
                  : 'Trigger a manual retraining run to recalibrate XGBoost and PyG weights.'}
              </p>
            </div>
          </div>
          <button
            type="button"
            disabled={busy || isRunningJob}
            onClick={handleManualRetrain}
            className="px-3.5 py-2 text-xs font-medium text-white bg-red-600 hover:bg-red-700 disabled:opacity-50 rounded-md whitespace-nowrap shrink-0"
          >
            {isRunningJob ? 'Retraining Running...' : 'Retrain Now'}
          </button>
        </div>
      )}

      {/* Charts Row: Distribution Comparison + PSI Timeline */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <section className="xl:col-span-6 bg-white border border-slate-200 rounded-lg p-5">
          <div className="pb-4 border-b border-slate-200 flex items-center justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold text-slate-900">
                Reference vs. Live Distribution — {selectedFeature?.displayName}
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                PSI <span className="font-mono tabular-nums">{selectedFeature?.psi.toFixed(3)}</span> · KS{' '}
                <span className="font-mono tabular-nums">{selectedFeature?.ksStat.toFixed(3)}</span> · p-value{' '}
                <span className="font-mono tabular-nums">{selectedFeature?.pValue.toFixed(4)}</span>
              </p>
            </div>
          </div>

          <div className="h-64 pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={selectedFeature?.bins ?? []} margin={{ top: 6, right: 12, left: -16, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                <XAxis
                  dataKey="bucket"
                  tick={{ fontSize: 11, fill: '#64748B', fontFamily: 'JetBrains Mono' }}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: '#64748B', fontFamily: 'JetBrains Mono' }}
                  unit="%"
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
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Bar dataKey="referencePct" name="Reference Baseline (%)" fill="#94A3B8" radius={[3, 3, 0, 0]} />
                <Bar
                  dataKey="currentPct"
                  name="Current Live Window (%)"
                  fill={selectedFeature?.drifted ? '#DC2626' : '#1D4ED8'}
                  radius={[3, 3, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="xl:col-span-6 bg-white border border-slate-200 rounded-lg p-5">
          <div className="pb-4 border-b border-slate-200 flex items-center justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold text-slate-900">
                Dataset PSI Trajectory Over Time
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Tracks continuous Population Stability Index against the retraining trigger line ({driftReport.driftThresholdPsi.toFixed(2)})
              </p>
            </div>
            <Activity className="w-4 h-4 text-slate-400 shrink-0" />
          </div>

          <div className="h-64 pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={driftReport.history} margin={{ top: 8, right: 14, left: -16, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 11, fill: '#64748B', fontFamily: 'JetBrains Mono' }}
                  minTickGap={24}
                />
                <YAxis
                  domain={[0, 0.45]}
                  tick={{ fontSize: 11, fill: '#64748B', fontFamily: 'JetBrains Mono' }}
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
                <ReferenceLine
                  y={driftReport.driftThresholdPsi}
                  stroke="#DC2626"
                  strokeDasharray="4 4"
                  label={{ value: 'Trigger', position: 'insideTopRight', fill: '#DC2626', fontSize: 11 }}
                />
                <Line
                  type="monotone"
                  dataKey="datasetPsi"
                  name="Dataset PSI"
                  stroke="#1D4ED8"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      {/* Feature Drift Breakdown + Retraining Jobs */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        <section className="xl:col-span-7 bg-white border border-slate-200 rounded-lg overflow-hidden">
          <div className="p-4 border-b border-slate-200">
            <h2 className="text-sm font-semibold text-slate-900">
              Evidently AI Feature Drift Report
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Click any row to inspect its reference vs. current histogram above.
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50/70 text-[11px] font-semibold text-slate-500">
                  <th className="py-2.5 px-4">Feast Feature</th>
                  <th className="py-2.5 px-3 text-right">Ref Mean</th>
                  <th className="py-2.5 px-3 text-right">Live Mean</th>
                  <th className="py-2.5 px-3 text-right">PSI</th>
                  <th className="py-2.5 px-3 text-right">KS Stat</th>
                  <th className="py-2.5 px-4 text-right">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-xs">
                {driftReport.features.map((f) => {
                  const isSelected = f.feature === selectedFeature?.feature;
                  return (
                    <tr
                      key={f.feature}
                      onClick={() => setSelectedFeatureKey(f.feature)}
                      className={`cursor-pointer transition-colors ${
                        isSelected ? 'bg-blue-50/70' : 'hover:bg-slate-50'
                      }`}
                    >
                      <td className="py-2.5 px-4">
                        <div className="font-medium text-slate-900">{f.displayName}</div>
                        <div className="text-[11px] font-mono text-slate-500">{f.feature}</div>
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-600">
                        {f.referenceMean.toFixed(2)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono tabular-nums font-medium text-slate-900">
                        {f.currentMean.toFixed(2)}
                      </td>
                      <td
                        className={`py-2.5 px-3 text-right font-mono tabular-nums font-semibold ${
                          f.drifted ? 'text-red-700' : 'text-slate-800'
                        }`}
                      >
                        {f.psi.toFixed(3)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-600">
                        {f.ksStat.toFixed(3)}
                      </td>
                      <td className="py-2.5 px-4 text-right whitespace-nowrap">
                        <span className={f.drifted ? 'text-red-700 font-semibold' : 'text-emerald-700 font-medium'}>
                          {f.drifted ? 'Drift Detected' : 'Nominal'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        {/* Automated Retraining Pipeline History */}
        <section className="xl:col-span-5 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between gap-3 pb-3 border-b border-slate-200">
            <div>
              <h2 className="text-sm font-semibold text-slate-900">Retraining Pipeline Jobs</h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Feast materialization → XGBoost + PyG fit → MLflow promotion
              </p>
            </div>
            <button
              type="button"
              disabled={busy || isRunningJob}
              onClick={handleManualRetrain}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-blue-700 hover:bg-blue-800 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap shrink-0"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRunningJob ? 'animate-spin' : ''}`} />
              Trigger Retrain
            </button>
          </div>

          <div className="divide-y divide-slate-100">
            {driftReport.jobs.map((job) => (
              <div key={job.id} className="py-3 first:pt-0 last:pb-0 space-y-1.5 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-mono font-semibold text-slate-900">{job.id}</span>
                  <span
                    className={
                      job.status === 'running'
                        ? 'text-blue-700 font-semibold'
                        : 'text-emerald-700 font-medium'
                    }
                  >
                    {job.status === 'running' ? `Running (${job.progressPct}%)` : 'Completed'}
                  </span>
                </div>
                <p className="text-slate-600">{job.reason}</p>
                {job.status === 'running' && (
                  <div className="h-1.5 bg-slate-100 rounded-xs overflow-hidden">
                    <div
                      className="h-full bg-blue-700 transition-all duration-300"
                      style={{ width: `${job.progressPct}%` }}
                    />
                  </div>
                )}
                <p className="text-[11px] text-slate-500 font-mono tabular-nums">
                  {job.triggeredAt} · {job.feedbackSamplesUsed} feedback labels
                  {job.resultingRunId ? ` · Registered ${job.resultingRunId}` : ''}
                </p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
