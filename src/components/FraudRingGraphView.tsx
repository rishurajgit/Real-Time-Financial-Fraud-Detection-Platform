import { useMemo, useState } from 'react';
import { Lock, ShieldAlert, Unlock } from 'lucide-react';
import type { FraudRing, GraphEntityType, GraphNode } from '../types/fraud';

interface FraudRingGraphViewProps {
  rings: FraudRing[];
  initialRingId: string | null;
  onRingAction: (ringId: string, action: 'freeze' | 'escalate' | 'release') => Promise<void>;
}

interface PositionedNode extends GraphNode {
  x: number;
  y: number;
}

function formatUsd(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(amount);
}

function nodeColors(type: GraphEntityType, frozen: boolean): { fill: string; stroke: string; text: string } {
  if (frozen) {
    return { fill: '#F1F5F9', stroke: '#64748B', text: '#475569' };
  }
  switch (type) {
    case 'account':
      return { fill: '#FEF2F2', stroke: '#DC2626', text: '#991B1B' };
    case 'device':
      return { fill: '#EFF6FF', stroke: '#1D4ED8', text: '#1E40AF' };
    case 'card':
      return { fill: '#FFFBEB', stroke: '#D97706', text: '#92400E' };
    case 'ip':
      return { fill: '#F8FAFC', stroke: '#475569', text: '#0F172A' };
  }
}

function entityTypeName(type: GraphEntityType): string {
  if (type === 'account') return 'Account';
  if (type === 'device') return 'Shared Device';
  if (type === 'card') return 'Shared Card BIN';
  return 'Proxy / Exit IP';
}

export function FraudRingGraphView({
  rings,
  initialRingId,
  onRingAction,
}: FraudRingGraphViewProps) {
  const [selectedRingId, setSelectedRingId] = useState<string>(initialRingId ?? rings[0]?.id ?? '');
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [actionBusy, setActionBusy] = useState(false);

  const activeRing = useMemo(() => {
    return rings.find((r) => r.id === selectedRingId) ?? rings[0] ?? null;
  }, [rings, selectedRingId]);

  const layout = useMemo(() => {
    if (!activeRing) return { nodes: [] as PositionedNode[], nodeMap: new Map<string, PositionedNode>() };
    const accounts = activeRing.nodes.filter((n) => n.type === 'account');
    const shared = activeRing.nodes.filter((n) => n.type !== 'account');

    const positioned: PositionedNode[] = [];
    const width = 680;
    const height = 420;

    // Place shared infrastructure nodes along central hub column
    shared.forEach((node, idx) => {
      const spacing = height / (shared.length + 1);
      positioned.push({
        ...node,
        x: Math.round(width * 0.5),
        y: Math.round(spacing * (idx + 1)),
      });
    });

    // Place linked accounts symmetrically on left and right arcs
    accounts.forEach((node, idx) => {
      const isLeft = idx % 2 === 0;
      const sideIndex = Math.floor(idx / 2);
      const sideCount = Math.ceil(accounts.length / 2);
      const spacing = height / (sideCount + 1);
      positioned.push({
        ...node,
        x: isLeft ? 130 : width - 130,
        y: Math.round(spacing * (sideIndex + 1) + (isLeft ? 0 : 12)),
      });
    });

    const nodeMap = new Map<string, PositionedNode>();
    positioned.forEach((n) => nodeMap.set(n.id, n));
    return { nodes: positioned, nodeMap };
  }, [activeRing]);

  const selectedNode = useMemo(() => {
    if (!activeRing) return null;
    return activeRing.nodes.find((n) => n.id === selectedNodeId) ?? activeRing.nodes[0] ?? null;
  }, [activeRing, selectedNodeId]);

  const handleAction = async (action: 'freeze' | 'escalate' | 'release') => {
    if (!activeRing) return;
    setActionBusy(true);
    try {
      await onRingAction(activeRing.id, action);
    } finally {
      setActionBusy(false);
    }
  };

  if (!activeRing) {
    return (
      <div className="bg-white border border-slate-200 rounded-lg p-12 text-center text-sm text-slate-500">
        No connected fraud rings detected in the current PyTorch Geometric graph window.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Ring Selector Row */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {rings.map((ring) => {
          const isSelected = ring.id === activeRing.id;
          return (
            <button
              key={ring.id}
              type="button"
              onClick={() => {
                setSelectedRingId(ring.id);
                setSelectedNodeId(null);
              }}
              className={`text-left p-5 rounded-lg border transition-colors bg-white ${
                isSelected
                  ? 'border-blue-600 ring-1 ring-blue-600'
                  : 'border-slate-200 hover:border-slate-300'
              }`}
            >
              <div className="flex items-center justify-between text-xs">
                <span className="font-mono font-semibold text-slate-900">{ring.id}</span>
                <span
                  className={
                    ring.status === 'frozen'
                      ? 'text-slate-600 font-medium'
                      : ring.status === 'escalated'
                        ? 'text-amber-700 font-semibold'
                        : 'text-red-700 font-semibold'
                  }
                >
                  {ring.status === 'frozen'
                    ? 'Frozen'
                    : ring.status === 'escalated'
                      ? 'Escalated'
                      : 'Active Ring'}
                </span>
              </div>
              <h3 className="mt-1.5 text-sm font-semibold text-slate-900">{ring.name}</h3>
              <p className="mt-1 text-xs text-slate-500 line-clamp-2">{ring.pattern}</p>
              <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-xs font-mono tabular-nums text-slate-700">
                <span>GNN {Math.round(ring.gnnConfidence * 100)}%</span>
                <span>{ring.accountCount} accts</span>
                <span>{formatUsd(ring.totalExposure)}</span>
              </div>
            </button>
          );
        })}
      </section>

      {/* Main Graph Viewport + Entity Inspector */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        <section className="xl:col-span-8 bg-white border border-slate-200 rounded-lg overflow-hidden">
          <div className="p-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold text-slate-900">
                {activeRing.id} — {activeRing.name}
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                PyTorch Geometric 2-hop GraphSAGE subgraph · Click any node to inspect entity risk
              </p>
            </div>
            <div className="flex items-center gap-4 text-[11px] text-slate-600 shrink-0">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-red-600 inline-block" /> Account
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-blue-700 inline-block" /> Device
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-amber-600 inline-block" /> Card
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-slate-600 inline-block" /> IP
              </span>
            </div>
          </div>

          <div className="bg-slate-50/60 p-4">
            <svg
              viewBox="0 0 680 420"
              className="w-full h-[380px] select-none"
              role="img"
              aria-label={`Entity relationship graph for ${activeRing.name}`}
            >
              {/* Edges */}
              {activeRing.edges.map((edge) => {
                const src = layout.nodeMap.get(edge.source);
                const dst = layout.nodeMap.get(edge.target);
                if (!src || !dst) return null;
                const highlighted = selectedNode?.id === src.id || selectedNode?.id === dst.id;
                return (
                  <g key={edge.id}>
                    <line
                      x1={src.x}
                      y1={src.y}
                      x2={dst.x}
                      y2={dst.y}
                      stroke={highlighted ? '#1D4ED8' : '#CBD5E1'}
                      strokeWidth={highlighted ? 2.2 : 1.3}
                      strokeDasharray={edge.relation === 'originates_ip' ? '4 3' : undefined}
                    />
                  </g>
                );
              })}

              {/* Nodes */}
              {layout.nodes.map((node) => {
                const palette = nodeColors(node.type, node.frozen);
                const isSelected = selectedNode?.id === node.id;
                const isAccount = node.type === 'account';
                return (
                  <g
                    key={node.id}
                    transform={`translate(${node.x}, ${node.y})`}
                    onClick={() => setSelectedNodeId(node.id)}
                    className="cursor-pointer"
                  >
                    {isSelected && (
                      <circle
                        r={isAccount ? 30 : 28}
                        fill="none"
                        stroke="#1D4ED8"
                        strokeWidth={2}
                        strokeDasharray="3 2"
                      />
                    )}
                    {isAccount ? (
                      <circle
                        r={22}
                        fill={palette.fill}
                        stroke={palette.stroke}
                        strokeWidth={2}
                      />
                    ) : (
                      <rect
                        x={-22}
                        y={-18}
                        width={44}
                        height={36}
                        rx={6}
                        fill={palette.fill}
                        stroke={palette.stroke}
                        strokeWidth={2}
                      />
                    )}
                    <text
                      textAnchor="middle"
                      dy="4"
                      fontSize="10"
                      fontWeight="600"
                      fontFamily="JetBrains Mono, monospace"
                      fill={palette.text}
                    >
                      {Math.round(node.riskScore * 100)}
                    </text>
                    <text
                      textAnchor="middle"
                      dy={isAccount ? 36 : 33}
                      fontSize="10.5"
                      fontWeight="500"
                      fontFamily="JetBrains Mono, monospace"
                      fill="#0F172A"
                    >
                      {node.label}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </section>

        {/* Right Ring Controls & Selected Entity Panel */}
        <aside className="xl:col-span-4 bg-white border border-slate-200 rounded-lg p-5 space-y-5">
          <div className="pb-4 border-b border-slate-200">
            <p className="text-xs text-slate-500">Ring Enforcement Controls</p>
            <h3 className="mt-1 text-base font-semibold text-slate-900">{activeRing.name}</h3>
            <p className="mt-1 text-xs text-slate-600">{activeRing.pattern}</p>

            <dl className="grid grid-cols-2 gap-3 mt-4 text-xs">
              <div className="p-2.5 bg-slate-50 rounded-md">
                <dt className="text-slate-500">GNN Confidence</dt>
                <dd className="mt-0.5 font-mono tabular-nums font-semibold text-red-700">
                  {(activeRing.gnnConfidence * 100).toFixed(1)}%
                </dd>
              </div>
              <div className="p-2.5 bg-slate-50 rounded-md">
                <dt className="text-slate-500">Total Exposure</dt>
                <dd className="mt-0.5 font-mono tabular-nums font-semibold text-slate-900">
                  {formatUsd(activeRing.totalExposure)}
                </dd>
              </div>
            </dl>

            <div className="mt-4 flex flex-col gap-2">
              {activeRing.status !== 'frozen' ? (
                <button
                  type="button"
                  disabled={actionBusy}
                  onClick={() => handleAction('freeze')}
                  className="w-full flex items-center justify-center gap-1.5 px-3.5 py-2 text-xs font-medium text-white bg-red-600 hover:bg-red-700 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
                >
                  <Lock className="w-3.5 h-3.5" />
                  Freeze All Ring Accounts & Confirm Cases
                </button>
              ) : (
                <button
                  type="button"
                  disabled={actionBusy}
                  onClick={() => handleAction('release')}
                  className="w-full flex items-center justify-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
                >
                  <Unlock className="w-3.5 h-3.5" />
                  Release Account Freeze
                </button>
              )}

              {activeRing.status === 'active' && (
                <button
                  type="button"
                  disabled={actionBusy}
                  onClick={() => handleAction('escalate')}
                  className="w-full flex items-center justify-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-700 border border-slate-200 hover:bg-slate-50 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
                >
                  <ShieldAlert className="w-3.5 h-3.5" />
                  Escalate Ring to Senior Fraud Ops
                </button>
              )}
            </div>
          </div>

          {selectedNode && (
            <div className="space-y-2.5 text-xs">
              <p className="font-semibold text-slate-900">Selected Graph Entity</p>
              <div className="flex items-center justify-between">
                <span className="font-mono font-semibold text-slate-900">{selectedNode.label}</span>
                <span className="text-slate-500">{entityTypeName(selectedNode.type)}</span>
              </div>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 pt-1">
                <div className="flex justify-between">
                  <dt className="text-slate-500">Node Risk</dt>
                  <dd className="font-mono tabular-nums font-medium text-slate-900">
                    {selectedNode.riskScore.toFixed(2)}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-500">State</dt>
                  <dd className={selectedNode.frozen ? 'text-slate-600 font-medium' : 'text-red-700 font-medium'}>
                    {selectedNode.frozen ? 'Frozen' : 'Active'}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-500">Linked TXs</dt>
                  <dd className="font-mono tabular-nums text-slate-900">{selectedNode.txCount}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-500">Exposure</dt>
                  <dd className="font-mono tabular-nums text-slate-900">
                    {formatUsd(selectedNode.totalAmount)}
                  </dd>
                </div>
              </dl>
            </div>
          )}

          <div className="pt-4 border-t border-slate-200 space-y-2 text-xs">
            <p className="font-semibold text-slate-900">Linked Open Cases ({activeRing.linkedCaseIds.length})</p>
            {activeRing.linkedCaseIds.length === 0 ? (
              <p className="text-slate-500">No linked cases generated yet in this session.</p>
            ) : (
              <div className="flex flex-wrap gap-x-3 gap-y-1 font-mono text-slate-700">
                {activeRing.linkedCaseIds.slice(0, 8).map((cid) => (
                  <span key={cid}>{cid}</span>
                ))}
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
