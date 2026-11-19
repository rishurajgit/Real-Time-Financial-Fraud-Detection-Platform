import { useCallback, useEffect, useState } from 'react';
import type {
  AnalystCase,
  CaseStatus,
  DriftReport,
  FeedbackLabel,
  ModelConfig,
  PlatformMetrics,
  PlatformStateSnapshot,
  ScoreTransactionRequest,
  ScoredTransaction,
  StreamControlState,
  ThroughputPoint,
} from '../types/fraud';

interface StreamTxEvent {
  transaction: ScoredTransaction;
  metrics: PlatformMetrics;
  driftReport: DriftReport;
  throughputPoint?: ThroughputPoint;
  newCase?: AnalystCase | null;
}

export function useFraudPlatform() {
  const [state, setState] = useState<PlatformStateSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [streamConnected, setStreamConnected] = useState(false);

  const fetchState = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/state');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as PlatformStateSnapshot;
      setState(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to reach scoring service');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchState();

    const eventSource = new EventSource('/api/v1/stream');

    eventSource.onopen = () => {
      setStreamConnected(true);
    };

    eventSource.onerror = () => {
      setStreamConnected(false);
    };

    eventSource.addEventListener('transaction', (evt) => {
      try {
        const payload = JSON.parse((evt as MessageEvent).data) as StreamTxEvent;
        setState((prev) => {
          if (!prev) return prev;
          const nextTransactions = [payload.transaction, ...prev.transactions].slice(0, 80);
          const nextCases =
            payload.newCase && !prev.cases.some((c) => c.id === payload.newCase!.id)
              ? [payload.newCase, ...prev.cases].slice(0, 60)
              : prev.cases;
          const nextThroughput = payload.throughputPoint
            ? [...prev.throughputHistory.slice(-23), payload.throughputPoint]
            : prev.throughputHistory;

          return {
            ...prev,
            metrics: payload.metrics,
            driftReport: payload.driftReport,
            transactions: nextTransactions,
            cases: nextCases,
            throughputHistory: nextThroughput,
          };
        });
      } catch {
        // Ignore malformed frame
      }
    });

    eventSource.addEventListener('state_sync', (evt) => {
      try {
        const snapshot = JSON.parse((evt as MessageEvent).data) as PlatformStateSnapshot;
        setState(snapshot);
      } catch {
        // Ignore malformed frame
      }
    });

    return () => {
      eventSource.close();
    };
  }, [fetchState]);

  const updateStreamControl = useCallback(async (patch: Partial<StreamControlState>) => {
    await fetch('/api/v1/stream/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
  }, []);

  const injectFraudBurst = useCallback(async () => {
    await fetch('/api/v1/stream/burst', { method: 'POST' });
  }, []);

  const updateCase = useCallback(
    async (
      caseId: string,
      patch: { status?: CaseStatus; feedbackLabel?: FeedbackLabel; notes?: string }
    ) => {
      const res = await fetch(`/api/v1/cases/${encodeURIComponent(caseId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      });
      if (res.ok) {
        await fetchState();
      }
    },
    [fetchState]
  );

  const performRingAction = useCallback(
    async (ringId: string, action: 'freeze' | 'escalate' | 'release') => {
      const res = await fetch(`/api/v1/rings/${encodeURIComponent(ringId)}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
      if (res.ok) {
        await fetchState();
      }
    },
    [fetchState]
  );

  const updateModelConfig = useCallback(async (patch: Partial<ModelConfig>) => {
    await fetch('/api/v1/model/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
  }, []);

  const updateDriftConfig = useCallback(
    async (patch: { autoRetrainEnabled?: boolean; driftThresholdPsi?: number }) => {
      await fetch('/api/v1/drift/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      });
    },
    []
  );

  const triggerRetraining = useCallback(async (reason?: string) => {
    await fetch('/api/v1/retrain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason }),
    });
  }, []);

  const scoreCustomTransaction = useCallback(async (req: ScoreTransactionRequest) => {
    const res = await fetch('/api/v1/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      throw new Error(`Scoring request failed (${res.status})`);
    }
    return (await res.json()) as ScoredTransaction;
  }, []);

  return {
    state,
    loading,
    error,
    streamConnected,
    refresh: fetchState,
    updateStreamControl,
    injectFraudBurst,
    updateCase,
    performRingAction,
    updateModelConfig,
    updateDriftConfig,
    triggerRetraining,
    scoreCustomTransaction,
  };
}
