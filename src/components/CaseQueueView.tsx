import { useMemo, useState } from 'react';
import { ArrowUpRight, Check, Search, ShieldAlert } from 'lucide-react';
import type { AnalystCase, CaseStatus, FeedbackLabel } from '../types/fraud';

interface CaseQueueViewProps {
  cases: AnalystCase[];
  feedbackLabelsCollected: number;
  onUpdateCase: (
    caseId: string,
    patch: { status?: CaseStatus; feedbackLabel?: FeedbackLabel; notes?: string }
  ) => Promise<void>;
  onOpenRing: (ringId: string) => void;
}

const STATUS_TABS: Array<{ id: 'all' | CaseStatus; label: string }> = [
  { id: 'all', label: 'All Cases' },
  { id: 'open', label: 'Open' },
  { id: 'investigating', label: 'Investigating' },
  { id: 'confirmed_fraud', label: 'Confirmed Fraud' },
  { id: 'false_positive', label: 'False Positive' },
];

function formatUsd(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(amount);
}

function statusMeta(status: CaseStatus): { label: string; className: string } {
  switch (status) {
    case 'open':
      return { label: 'Open', className: 'text-amber-700 font-semibold' };
    case 'investigating':
      return { label: 'Investigating', className: 'text-blue-700 font-semibold' };
    case 'confirmed_fraud':
      return { label: 'Confirmed Fraud', className: 'text-red-700 font-semibold' };
    case 'false_positive':
      return { label: 'False Positive', className: 'text-emerald-700 font-medium' };
  }
}

export function CaseQueueView({
  cases,
  feedbackLabelsCollected,
  onUpdateCase,
  onOpenRing,
}: CaseQueueViewProps) {
  const [statusFilter, setStatusFilter] = useState<'all' | CaseStatus>('all');
  const [search, setSearch] = useState('');
  const [selectedId, setSelectedId] = useState<string | null>(cases[0]?.id ?? null);
  const [draftNotes, setDraftNotes] = useState('');
  const [savedNotice, setSavedNotice] = useState<string | null>(null);

  const filteredCases = useMemo(() => {
    const q = search.trim().toLowerCase();
    return cases.filter((c) => {
      if (statusFilter !== 'all' && c.status !== statusFilter) return false;
      if (!q) return true;
      return (
        c.id.toLowerCase().includes(q) ||
        c.accountId.toLowerCase().includes(q) ||
        c.merchantName.toLowerCase().includes(q) ||
        c.topReason.toLowerCase().includes(q) ||
        (c.ringId?.toLowerCase().includes(q) ?? false)
      );
    });
  }, [cases, statusFilter, search]);

  const activeCase = useMemo(() => {
    const found = cases.find((c) => c.id === selectedId);
    return found ?? filteredCases[0] ?? null;
  }, [cases, selectedId, filteredCases]);

  const selectCase = (c: AnalystCase) => {
    setSelectedId(c.id);
    setDraftNotes(c.notes);
    setSavedNotice(null);
  };

  const applyDecision = async (status: CaseStatus, feedbackLabel?: FeedbackLabel) => {
    if (!activeCase) return;
    await onUpdateCase(activeCase.id, {
      status,
      feedbackLabel: feedbackLabel ?? activeCase.feedbackLabel,
      notes: draftNotes || activeCase.notes,
    });
    setSavedNotice(`Updated ${activeCase.id} to ${statusMeta(status).label}.`);
    setTimeout(() => setSavedNotice(null), 3500);
  };

  const openCount = cases.filter((c) => c.status === 'open').length;
  const investigatingCount = cases.filter((c) => c.status === 'investigating').length;
  const confirmedCount = cases.filter((c) => c.status === 'confirmed_fraud').length;
  const fpCount = cases.filter((c) => c.status === 'false_positive').length;
  const precisionLive =
    confirmedCount + fpCount > 0 ? Math.round((confirmedCount / (confirmedCount + fpCount)) * 100) : 84;

  return (
    <div className="space-y-6">
      {/* Summary Strip */}
      <section className="bg-white border border-slate-200 rounded-lg">
        <div className="grid grid-cols-2 lg:grid-cols-4 divide-y lg:divide-y-0 sm:divide-x divide-slate-200">
          <div className="p-5">
            <p className="text-xs text-slate-500">Open / Investigating</p>
            <p className="mt-1.5 text-2xl font-semibold font-mono tabular-nums text-slate-900">
              {openCount} <span className="text-base font-normal text-slate-400">/ {investigatingCount}</span>
            </p>
            <p className="mt-1 text-xs text-slate-500">Awaiting analyst adjudication</p>
          </div>
          <div className="p-5">
            <p className="text-xs text-slate-500">Confirmed Fraud</p>
            <p className="mt-1.5 text-2xl font-semibold font-mono tabular-nums text-red-700">
              {confirmedCount}
            </p>
            <p className="mt-1 text-xs text-slate-500">Positive labels fed to Feast</p>
          </div>
          <div className="p-5">
            <p className="text-xs text-slate-500">Analyst Precision</p>
            <p className="mt-1.5 text-2xl font-semibold font-mono tabular-nums text-slate-900">
              {precisionLive}%
            </p>
            <p className="mt-1 text-xs text-slate-500">
              <span className="font-mono tabular-nums">{fpCount}</span> false positives cleared
            </p>
          </div>
          <div className="p-5">
            <p className="text-xs text-slate-500">Retraining Feedback Store</p>
            <p className="mt-1.5 text-2xl font-semibold font-mono tabular-nums text-slate-900">
              {feedbackLabelsCollected}
            </p>
            <p className="mt-1 text-xs text-slate-500">Ground-truth labels ready</p>
          </div>
        </div>
      </section>

      {/* Queue + Investigation Workspace */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        <section className="xl:col-span-8 bg-white border border-slate-200 rounded-lg overflow-hidden">
          <div className="p-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md overflow-x-auto">
              {STATUS_TABS.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setStatusFilter(tab.id)}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                    statusFilter === tab.id
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <div className="relative w-full sm:w-56">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Filter cases..."
                className="w-full pl-8 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-md text-slate-900 placeholder:text-slate-400 focus:outline-none focus:bg-white focus:border-blue-600"
              />
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50/70 text-[11px] font-semibold text-slate-500">
                  <th className="py-2.5 px-4">Case / Account</th>
                  <th className="py-2.5 px-3">Merchant & Primary Signal</th>
                  <th className="py-2.5 px-3 text-right">Amount</th>
                  <th className="py-2.5 px-3 text-right">Score</th>
                  <th className="py-2.5 px-4 text-right">Status / Label</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-xs">
                {filteredCases.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-12 text-center text-slate-500">
                      No cases in this status view.
                    </td>
                  </tr>
                ) : (
                  filteredCases.map((c) => {
                    const st = statusMeta(c.status);
                    const isSelected = activeCase?.id === c.id;
                    return (
                      <tr
                        key={c.id}
                        onClick={() => selectCase(c)}
                        className={`cursor-pointer transition-colors ${
                          isSelected ? 'bg-blue-50/70' : 'hover:bg-slate-50'
                        }`}
                      >
                        <td className="py-2.5 px-4 whitespace-nowrap">
                          <div className="font-mono font-medium text-slate-900 tabular-nums">{c.id}</div>
                          <div className="text-[11px] font-mono text-slate-500">
                            {c.accountId}
                            {c.ringId ? ` · ${c.ringId}` : ''}
                          </div>
                        </td>
                        <td className="py-2.5 px-3 max-w-[230px]">
                          <div className="font-medium text-slate-900 truncate">{c.merchantName}</div>
                          <div className="text-[11px] text-slate-500 truncate">{c.topReason}</div>
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums font-medium text-slate-900 whitespace-nowrap">
                          {formatUsd(c.amount)}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-900">
                          {c.blendedScore.toFixed(3)}
                        </td>
                        <td className="py-2.5 px-4 text-right whitespace-nowrap">
                          <div className={st.className}>{st.label}</div>
                          <div className="text-[11px] text-slate-400">
                            Label: {c.feedbackLabel}
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

        {/* Case Detail & Feedback Panel */}
        <aside className="xl:col-span-4 bg-white border border-slate-200 rounded-lg p-5">
          {!activeCase ? (
            <div className="py-12 text-center text-xs text-slate-500">
              Select a case to review evidence and submit training labels.
            </div>
          ) : (
            <div className="space-y-5">
              <div className="pb-4 border-b border-slate-200">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs text-slate-500">
                    {activeCase.id} · {activeCase.transactionId}
                  </span>
                  <span className={`text-xs ${statusMeta(activeCase.status).className}`}>
                    {statusMeta(activeCase.status).label}
                  </span>
                </div>
                <h3 className="mt-1 text-base font-semibold text-slate-900">
                  {activeCase.merchantName}
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Account {activeCase.accountId} · Assigned to {activeCase.assignee}
                </p>
              </div>

              <dl className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 bg-slate-50 rounded-md">
                  <dt className="text-slate-500">Exposure Amount</dt>
                  <dd className="mt-1 text-sm font-semibold font-mono tabular-nums text-slate-900">
                    {formatUsd(activeCase.amount)}
                  </dd>
                </div>
                <div className="p-3 bg-slate-50 rounded-md">
                  <dt className="text-slate-500">Blended Risk Score</dt>
                  <dd className="mt-1 text-sm font-semibold font-mono tabular-nums text-red-700">
                    {activeCase.blendedScore.toFixed(3)}
                  </dd>
                </div>
                <div className="p-3 bg-slate-50 rounded-md">
                  <dt className="text-slate-500">XGBoost Sub-score</dt>
                  <dd className="mt-1 text-sm font-semibold font-mono tabular-nums text-slate-900">
                    {activeCase.xgbScore.toFixed(3)}
                  </dd>
                </div>
                <div className="p-3 bg-slate-50 rounded-md">
                  <dt className="text-slate-500">PyG GNN Sub-score</dt>
                  <dd className="mt-1 text-sm font-semibold font-mono tabular-nums text-slate-900">
                    {activeCase.gnnScore.toFixed(3)}
                  </dd>
                </div>
              </dl>

              <div className="space-y-1.5 text-xs">
                <p className="font-medium text-slate-700">Primary Alert Driver</p>
                <p className="text-slate-600">{activeCase.topReason}</p>
              </div>

              {activeCase.ringId && (
                <button
                  type="button"
                  onClick={() => onOpenRing(activeCase.ringId!)}
                  className="w-full flex items-center justify-between px-3.5 py-2 text-xs font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 rounded-md transition-colors whitespace-nowrap"
                >
                  <span>Inspect Connected Ring ({activeCase.ringId})</span>
                  <ArrowUpRight className="w-3.5 h-3.5" />
                </button>
              )}

              <div className="pt-4 border-t border-slate-200 space-y-2">
                <label htmlFor="case-notes" className="block text-xs font-medium text-slate-700">
                  Analyst Investigation Notes
                </label>
                <textarea
                  id="case-notes"
                  rows={3}
                  value={draftNotes}
                  onChange={(e) => setDraftNotes(e.target.value)}
                  placeholder="Document device correlation, cardholder callback result, or ring evidence..."
                  className="w-full px-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded-md text-slate-900 placeholder:text-slate-400 focus:outline-none focus:bg-white focus:border-blue-600"
                />
              </div>

              <div className="space-y-2">
                <p className="text-xs font-medium text-slate-700">Adjudicate & Submit Ground-Truth Label</p>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => applyDecision('confirmed_fraud', 'fraud')}
                    className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-white bg-red-600 hover:bg-red-700 rounded-md transition-colors whitespace-nowrap"
                  >
                    <ShieldAlert className="w-3.5 h-3.5" />
                    Confirm Fraud
                  </button>
                  <button
                    type="button"
                    onClick={() => applyDecision('false_positive', 'legitimate')}
                    className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200 rounded-md transition-colors whitespace-nowrap"
                  >
                    <Check className="w-3.5 h-3.5" />
                    False Positive
                  </button>
                </div>
                <button
                  type="button"
                  onClick={() => applyDecision('investigating')}
                  className="w-full px-3 py-2 text-xs font-medium text-slate-700 border border-slate-200 hover:bg-slate-50 rounded-md transition-colors whitespace-nowrap"
                >
                  Save Notes & Mark In Review
                </button>
              </div>

              {savedNotice && (
                <p className="text-xs text-emerald-700 bg-emerald-50 px-3 py-2 rounded-md">
                  {savedNotice}
                </p>
              )}
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
