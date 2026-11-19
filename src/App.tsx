import { useState } from 'react';
import {
  Activity,
  BarChart3,
  Briefcase,
  Cpu,
  Menu,
  Network,
  Pause,
  Play,
  Radio,
  ShieldAlert,
  X,
} from 'lucide-react';
import { useFraudPlatform } from './hooks/useFraudPlatform';
import { LiveStreamView } from './components/LiveStreamView';
import { CaseQueueView } from './components/CaseQueueView';
import { FraudRingGraphView } from './components/FraudRingGraphView';
import { ModelMlflowView } from './components/ModelMlflowView';
import { DriftRetrainingView } from './components/DriftRetrainingView';
import { ArchitectureApiView } from './components/ArchitectureApiView';

type TabId = 'live' | 'cases' | 'graph' | 'models' | 'drift' | 'architecture';

interface NavItem {
  id: TabId;
  label: string;
  sectionLabel: string;
  icon: typeof Radio;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'live', label: 'Live Stream', sectionLabel: 'Live Stream & Triage', icon: Radio },
  { id: 'cases', label: 'Case Queue', sectionLabel: 'Analyst Case Queue', icon: Briefcase },
  { id: 'graph', label: 'Fraud Rings', sectionLabel: 'PyG Fraud Ring Graph', icon: Network },
  { id: 'models', label: 'Model & MLflow', sectionLabel: 'PR-AUC & MLflow Registry', icon: BarChart3 },
  { id: 'drift', label: 'Drift & Retrain', sectionLabel: 'Evidently AI Drift Monitor', icon: Activity },
  { id: 'architecture', label: 'System & API', sectionLabel: 'Architecture & Scoring API', icon: Cpu },
];

const ANALYST_AVATAR_SRC = '/src/assets/images/avatar_risk_analyst_1795081528475.jpg';

export default function App() {
  const {
    state,
    loading,
    error,
    streamConnected,
    refresh,
    updateStreamControl,
    injectFraudBurst,
    updateCase,
    performRingAction,
    updateModelConfig,
    updateDriftConfig,
    triggerRetraining,
    scoreCustomTransaction,
  } = useFraudPlatform();

  const [activeTab, setActiveTab] = useState<TabId>('live');
  const [focusedRingId, setFocusedRingId] = useState<string | null>(null);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [avatarFailed, setAvatarFailed] = useState(false);

  const activeNav = NAV_ITEMS.find((n) => n.id === activeTab) ?? NAV_ITEMS[0];

  const handleOpenRing = (ringId: string) => {
    setFocusedRingId(ringId);
    setActiveTab('graph');
  };

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-[#0F172A] flex">
      {/* Desktop Left Sidebar */}
      <aside className="hidden lg:flex lg:flex-col w-60 shrink-0 bg-white border-r border-slate-200 sticky top-0 h-screen">
        <div className="h-14 px-5 flex items-center border-b border-slate-200">
          <button
            type="button"
            onClick={() => setActiveTab('live')}
            className="text-base font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0"
          >
            SentinelRisk
          </button>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto" aria-label="Primary navigation">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            const countText =
              item.id === 'cases' && state
                ? String(state.metrics.openCases)
                : item.id === 'graph' && state
                  ? String(state.metrics.activeRings)
                  : item.id === 'drift' && state?.driftReport.driftAlertActive
                    ? 'Alert'
                    : null;

            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center justify-between gap-2.5 px-3 py-2 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                  isActive
                    ? 'bg-blue-50 text-blue-800 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                <span className="flex items-center gap-2.5 truncate">
                  <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-blue-700' : 'text-slate-400'}`} />
                  <span className="truncate">{item.label}</span>
                </span>
                {countText && (
                  <span
                    className={`font-mono tabular-nums text-[11px] shrink-0 ${
                      countText === 'Alert' ? 'text-red-700 font-semibold' : 'text-slate-500'
                    }`}
                  >
                    {countText}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Analyst Profile Footer */}
        <div className="p-4 border-t border-slate-200">
          <div className="flex items-center gap-3">
            {avatarFailed ? (
              <div className="w-8 h-8 rounded-full bg-slate-200 text-slate-700 flex items-center justify-center text-xs font-semibold shrink-0">
                EL
              </div>
            ) : (
              <img
                src={ANALYST_AVATAR_SRC}
                alt="Elena Lin, Lead Fraud Analyst"
                referrerPolicy="no-referrer"
                onError={() => setAvatarFailed(true)}
                className="w-8 h-8 rounded-full object-cover border border-slate-200 shrink-0"
              />
            )}
            <div className="min-w-0">
              <p className="text-xs font-semibold text-slate-900 truncate">Elena Lin</p>
              <p className="text-[11px] text-slate-500 truncate">Senior Risk Analyst</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Mobile Navigation Drawer */}
      {mobileNavOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-slate-900/40"
            onClick={() => setMobileNavOpen(false)}
            aria-hidden="true"
          />
          <div className="relative w-64 max-w-[80vw] bg-white h-full border-r border-slate-200 flex flex-col z-10">
            <div className="h-14 px-5 flex items-center justify-between border-b border-slate-200">
              <span className="text-base font-bold tracking-tight text-slate-900">SentinelRisk</span>
              <button
                type="button"
                onClick={() => setMobileNavOpen(false)}
                aria-label="Close navigation"
                className="p-1.5 text-slate-500 hover:text-slate-900 rounded"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <nav className="flex-1 px-3 py-4 space-y-1">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => {
                      setActiveTab(item.id);
                      setMobileNavOpen(false);
                    }}
                    className={`w-full flex items-center gap-2.5 px-3 py-2.5 text-xs font-medium rounded-md whitespace-nowrap ${
                      isActive
                        ? 'bg-blue-50 text-blue-800 font-semibold'
                        : 'text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>
        </div>
      )}

      {/* Main Content Workspace */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* Top Bar Contract: Breadcrumb Left + At Most 2 Crisp Primary Actions Right */}
        <header className="h-14 bg-white border-b border-slate-200 px-4 sm:px-6 flex items-center justify-between gap-4 sticky top-0 z-30">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setMobileNavOpen(true)}
              aria-label="Open navigation"
              className="lg:hidden p-1.5 -ml-1 text-slate-600 hover:text-slate-900 rounded"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-2 text-xs sm:text-sm min-w-0">
              <span className="text-slate-400 hidden sm:inline whitespace-nowrap">Fraud Ops</span>
              <span className="text-slate-300 hidden sm:inline" aria-hidden="true">/</span>
              <h1 className="font-semibold text-slate-900 truncate">{activeNav.sectionLabel}</h1>
              {state && (
                <span className="hidden md:inline text-xs text-slate-400 whitespace-nowrap ml-2">
                  · {streamConnected && state.stream.running ? 'Kafka Replay Active' : 'Stream Paused'}
                </span>
              )}
            </div>
          </div>

          {state && (
            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={() => updateStreamControl({ running: !state.stream.running })}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 border border-slate-200 hover:bg-slate-50 rounded-md transition-colors whitespace-nowrap shrink-0"
              >
                {state.stream.running ? (
                  <>
                    <Pause className="w-3.5 h-3.5" />
                    Pause Stream
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5" />
                    Resume Stream
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={injectFraudBurst}
                className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-medium text-white bg-blue-700 hover:bg-blue-800 rounded-md transition-colors whitespace-nowrap shrink-0"
              >
                <ShieldAlert className="w-3.5 h-3.5" />
                Inject Fraud Burst
              </button>
            </div>
          )}
        </header>

        {/* Main Viewport */}
        <main className="flex-1 p-4 sm:p-6 max-w-[1440px] w-full mx-auto">
          {loading ? (
            <div className="space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {[0, 1, 2, 3].map((i) => (
                  <div key={i} className="h-24 bg-white border border-slate-200 rounded-lg animate-pulse" />
                ))}
              </div>
              <div className="h-72 bg-white border border-slate-200 rounded-lg animate-pulse" />
            </div>
          ) : error || !state ? (
            <div className="bg-white border border-slate-200 rounded-lg p-8 text-center space-y-3">
              <p className="text-sm font-semibold text-slate-900">
                Unable to connect to the scoring engine
              </p>
              <p className="text-xs text-slate-500">{error ?? 'No snapshot received from server.'}</p>
              <button
                type="button"
                onClick={refresh}
                className="px-4 py-2 text-xs font-medium text-white bg-blue-700 hover:bg-blue-800 rounded-md"
              >
                Retry Connection
              </button>
            </div>
          ) : (
            <>
              {activeTab === 'live' && (
                <LiveStreamView
                  metrics={state.metrics}
                  stream={state.stream}
                  modelConfig={state.modelConfig}
                  transactions={state.transactions}
                  throughputHistory={state.throughputHistory}
                  onUpdateStream={updateStreamControl}
                  onUpdateCase={updateCase}
                  onOpenRing={handleOpenRing}
                />
              )}

              {activeTab === 'cases' && (
                <CaseQueueView
                  cases={state.cases}
                  feedbackLabelsCollected={state.metrics.feedbackLabelsCollected}
                  onUpdateCase={updateCase}
                  onOpenRing={handleOpenRing}
                />
              )}

              {activeTab === 'graph' && (
                <FraudRingGraphView
                  rings={state.rings}
                  initialRingId={focusedRingId}
                  onRingAction={performRingAction}
                />
              )}

              {activeTab === 'models' && (
                <ModelMlflowView
                  runs={state.mlflowRuns}
                  prCurve={state.prCurve}
                  modelConfig={state.modelConfig}
                  onUpdateModelConfig={updateModelConfig}
                />
              )}

              {activeTab === 'drift' && (
                <DriftRetrainingView
                  driftReport={state.driftReport}
                  stream={state.stream}
                  onToggleDriftInjection={(enable) => updateStreamControl({ driftInjected: enable })}
                  onUpdateDriftConfig={updateDriftConfig}
                  onTriggerRetrain={triggerRetraining}
                />
              )}

              {activeTab === 'architecture' && (
                <ArchitectureApiView
                  edaSummaries={state.edaSummaries}
                  activeDataset={state.stream.dataset}
                  onScoreTransaction={scoreCustomTransaction}
                />
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
