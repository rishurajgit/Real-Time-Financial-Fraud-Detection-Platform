import {
  AnalystCase,
  CaseStatus,
  DatasetEdaSummary,
  DatasetProfile,
  DecisionAction,
  DriftHistoryPoint,
  DriftReport,
  FeastFeatureVector,
  FeatureAttribution,
  FeatureDriftMetric,
  FeedbackLabel,
  FraudRing,
  GraphEdge,
  GraphNode,
  MlflowRun,
  ModelConfig,
  PlatformMetrics,
  PlatformStateSnapshot,
  PrCurvePoint,
  RetrainingJob,
  ScoreTransactionRequest,
  ScoredTransaction,
  StreamControlState,
  ThroughputPoint,
} from '../types/fraud.ts';

interface AccountBaseline {
  accountId: string;
  avgAmount: number;
  homeCity: string;
  primaryDevice: string;
  primaryCard: string;
  accountAgeDays: number;
  recentTimestamps: number[];
  ringId: string | null;
}

const CITIES = [
  'New York, US',
  'San Francisco, US',
  'Chicago, US',
  'Austin, US',
  'London, UK',
  'Toronto, CA',
  'Frankfurt, DE',
  'Singapore, SG',
  'Lagos, NG',
  'Bucharest, RO',
  'Sao Paulo, BR',
];

const MERCHANTS: Array<{ name: string; category: string; risk: number }> = [
  { name: 'Whole Foods Market', category: 'Grocery', risk: 0.08 },
  { name: 'Delta Air Lines', category: 'Travel', risk: 0.24 },
  { name: 'CloudScale Hosting', category: 'Digital Goods', risk: 0.42 },
  { name: 'Apex Electronics', category: 'Consumer Tech', risk: 0.48 },
  { name: 'Metro Transit Pass', category: 'Transit', risk: 0.05 },
  { name: 'Nordic Luxury Watches', category: 'Luxury Retail', risk: 0.66 },
  { name: 'FastRemit Wire', category: 'Money Transfer', risk: 0.78 },
  { name: 'BitVault Exchange', category: 'Crypto On-Ramp', risk: 0.84 },
  { name: 'Starbucks Coffee', category: 'Dining', risk: 0.06 },
  { name: 'Prime Gift Cards', category: 'Prepaid / Gift', risk: 0.74 },
  { name: 'Shell Fuel Station', category: 'Fuel', risk: 0.12 },
  { name: 'Global Freight Pay', category: 'B2B Wire', risk: 0.34 },
];

const PAYSIM_TYPES = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'];
const IEEE_TYPES = ['CARD_NOT_PRESENT', 'POS_CHIP', 'DIGITAL_WALLET', 'ACH_DEBIT', 'WIRE'];

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

function round(val: number, decimals = 2): number {
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

function formatShortTime(date: Date): string {
  return date.toTimeString().slice(0, 8);
}

function randomPick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

export class FraudPlatformEngine {
  private accounts = new Map<string, AccountBaseline>();
  private deviceToAccounts = new Map<string, Set<string>>();
  private cardToAccounts = new Map<string, Set<string>>();
  private transactions: ScoredTransaction[] = [];
  private cases: AnalystCase[] = [];
  private rings: FraudRing[] = [];
  private mlflowRuns: MlflowRun[] = [];
  private prCurve: PrCurvePoint[] = [];
  private driftHistory: DriftHistoryPoint[] = [];
  private retrainingJobs: RetrainingJob[] = [];
  private throughputHistory: ThroughputPoint[] = [];
  private latencies: number[] = [];
  private listeners = new Set<(event: string, payload: unknown) => void>();

  private totalProcessed = 14820;
  private flaggedCount = 196;
  private blockedCount = 128;
  private preventedLossUsd = 418920;
  private txCounter = 90400;
  private caseCounter = 4100;
  private jobCounter = 12;

  private streamControl: StreamControlState = {
    running: true,
    ratePerSec: 1.5,
    dataset: 'IEEE-CIS',
    driftInjected: false,
  };

  private modelConfig: ModelConfig = {
    activeRunId: 'mlf-run-048',
    xgbWeight: 0.65,
    gnnWeight: 0.35,
    reviewThreshold: 0.58,
    blockThreshold: 0.82,
  };

  private driftThresholdPsi = 0.2;
  private autoRetrainEnabled = true;
  private intervalHandle: ReturnType<typeof setInterval> | null = null;
  private lastRetrainTriggerMs = 0;

  constructor() {
    this.seedBaselinesAndRings();
    this.seedMlflowAndPrCurve();
    this.seedInitialHistory();
    this.startLoop();
  }

  public subscribe(listener: (event: string, payload: unknown) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private emit(event: string, payload: unknown): void {
    for (const listener of this.listeners) {
      listener(event, payload);
    }
  }

  private seedBaselinesAndRings(): void {
    // Seed 65 normal retail & commercial accounts
    for (let i = 1; i <= 65; i++) {
      const accountId = `ACC-${String(10000 + i * 17).slice(0, 5)}`;
      const primaryDevice = `DEV-${(1000 + i * 31).toString(16).toUpperCase()}`;
      const primaryCard = `CARD-****-${String(2100 + i * 43).slice(-4)}`;
      const baseline: AccountBaseline = {
        accountId,
        avgAmount: round(35 + (i % 9) * 48 + Math.random() * 60, 2),
        homeCity: CITIES[i % 7],
        primaryDevice,
        primaryCard,
        accountAgeDays: 90 + (i * 23) % 900,
        recentTimestamps: [],
        ringId: null,
      };
      this.accounts.set(accountId, baseline);
      this.registerEntityLink(this.deviceToAccounts, primaryDevice, accountId);
      this.registerEntityLink(this.cardToAccounts, primaryCard, accountId);
    }

    // Seed 3 realistic fraud rings linked by shared devices, cards, and proxy IPs
    const ringConfigs = [
      {
        id: 'RING-01',
        name: 'Velocity Card-Testing Cluster',
        pattern: '4 accounts sharing 1 emulated Android device & 2 rotating prepaid BINs',
        sharedDevice: 'DEV-F991A',
        sharedCard: 'CARD-****-8841',
        sharedIp: '185.220.101.44',
        gnnConfidence: 0.94,
        accountIds: ['ACC-88101', 'ACC-88102', 'ACC-88103', 'ACC-88104'],
        exposure: 64280,
      },
      {
        id: 'RING-02',
        name: 'Cross-Border Mule Cash-Out Mesh',
        pattern: '5 newly created mule accounts funneling PaySim transfers to 1 exit device',
        sharedDevice: 'DEV-C402B',
        sharedCard: 'CARD-****-3319',
        sharedIp: '91.134.208.17',
        gnnConfidence: 0.89,
        accountIds: ['ACC-77201', 'ACC-77202', 'ACC-77203', 'ACC-77204', 'ACC-77205'],
        exposure: 112450,
      },
      {
        id: 'RING-03',
        name: 'Luxury Retail Account Takeover Syndicate',
        pattern: '3 compromised high-limit accounts accessed via identical residential proxy',
        sharedDevice: 'DEV-A719E',
        sharedCard: 'CARD-****-9904',
        sharedIp: '45.142.214.92',
        gnnConfidence: 0.86,
        accountIds: ['ACC-66301', 'ACC-66302', 'ACC-66303'],
        exposure: 49800,
      },
    ];

    for (const cfg of ringConfigs) {
      const nodes: GraphNode[] = [];
      const edges: GraphEdge[] = [];

      nodes.push({
        id: cfg.sharedDevice,
        label: cfg.sharedDevice,
        type: 'device',
        riskScore: round(cfg.gnnConfidence - 0.02, 2),
        frozen: false,
        ringId: cfg.id,
        txCount: cfg.accountIds.length * 4,
        totalAmount: cfg.exposure,
      });
      nodes.push({
        id: cfg.sharedCard,
        label: cfg.sharedCard,
        type: 'card',
        riskScore: round(cfg.gnnConfidence - 0.05, 2),
        frozen: false,
        ringId: cfg.id,
        txCount: cfg.accountIds.length * 3,
        totalAmount: round(cfg.exposure * 0.72, 2),
      });
      nodes.push({
        id: cfg.sharedIp,
        label: cfg.sharedIp,
        type: 'ip',
        riskScore: round(cfg.gnnConfidence - 0.08, 2),
        frozen: false,
        ringId: cfg.id,
        txCount: cfg.accountIds.length * 5,
        totalAmount: cfg.exposure,
      });

      cfg.accountIds.forEach((accId, idx) => {
        const baseline: AccountBaseline = {
          accountId: accId,
          avgAmount: 190 + idx * 55,
          homeCity: CITIES[idx % 4],
          primaryDevice: cfg.sharedDevice,
          primaryCard: cfg.sharedCard,
          accountAgeDays: 4 + idx * 6,
          recentTimestamps: [],
          ringId: cfg.id,
        };
        this.accounts.set(accId, baseline);
        this.registerEntityLink(this.deviceToAccounts, cfg.sharedDevice, accId);
        this.registerEntityLink(this.cardToAccounts, cfg.sharedCard, accId);

        nodes.push({
          id: accId,
          label: accId,
          type: 'account',
          riskScore: round(clamp(cfg.gnnConfidence - idx * 0.03, 0.72, 0.98), 2),
          frozen: false,
          ringId: cfg.id,
          txCount: 3 + idx,
          totalAmount: round(cfg.exposure / cfg.accountIds.length, 2),
        });

        edges.push({
          id: `e-${accId}-dev`,
          source: accId,
          target: cfg.sharedDevice,
          relation: 'uses_device',
          weight: 0.92,
        });
        if (idx % 2 === 0 || idx === 1) {
          edges.push({
            id: `e-${accId}-card`,
            source: accId,
            target: cfg.sharedCard,
            relation: 'owns_card',
            weight: 0.85,
          });
        }
        edges.push({
          id: `e-${accId}-ip`,
          source: accId,
          target: cfg.sharedIp,
          relation: 'originates_ip',
          weight: 0.78,
        });
      });

      this.rings.push({
        id: cfg.id,
        name: cfg.name,
        detectedAt: new Date(Date.now() - 1000 * 60 * 35).toISOString(),
        gnnConfidence: cfg.gnnConfidence,
        accountCount: cfg.accountIds.length,
        sharedEntities: 3,
        totalExposure: cfg.exposure,
        status: 'active',
        pattern: cfg.pattern,
        nodes,
        edges,
        linkedCaseIds: [],
      });
    }
  }

  private registerEntityLink(map: Map<string, Set<string>>, key: string, accountId: string): void {
    if (!map.has(key)) {
      map.set(key, new Set());
    }
    map.get(key)!.add(accountId);
  }

  private seedMlflowAndPrCurve(): void {
    this.mlflowRuns = [
      {
        runId: 'mlf-run-048',
        name: 'ensemble-xgb-graphsage-v3',
        modelType: 'Blended Ensemble',
        createdAt: '2026-11-18 19:42 UTC',
        prAuc: 0.894,
        rocAuc: 0.986,
        f1AtThreshold: 0.842,
        precisionAt90Recall: 0.768,
        p99LatencyMs: 14.2,
        imbalanceStrategy: 'scale_pos_weight=112 + Focal GNN',
        blendWeightXgb: 0.65,
        blendWeightGnn: 0.35,
        status: 'production',
        dataset: 'IEEE-CIS',
      },
      {
        runId: 'mlf-run-047',
        name: 'pyg-graphsage-2hop-gat',
        modelType: 'PyG GraphSAGE',
        createdAt: '2026-11-18 16:15 UTC',
        prAuc: 0.812,
        rocAuc: 0.961,
        f1AtThreshold: 0.771,
        precisionAt90Recall: 0.642,
        p99LatencyMs: 8.6,
        imbalanceStrategy: 'Graph Focal Loss (alpha=0.92)',
        blendWeightXgb: 0.0,
        blendWeightGnn: 1.0,
        status: 'staging',
        dataset: 'IEEE-CIS',
      },
      {
        runId: 'mlf-run-045',
        name: 'xgb-feast-v2-classweight',
        modelType: 'XGBoost (Class Weights)',
        createdAt: '2026-11-17 22:08 UTC',
        prAuc: 0.851,
        rocAuc: 0.979,
        f1AtThreshold: 0.806,
        precisionAt90Recall: 0.712,
        p99LatencyMs: 5.4,
        imbalanceStrategy: 'scale_pos_weight=114',
        blendWeightXgb: 1.0,
        blendWeightGnn: 0.0,
        status: 'staging',
        dataset: 'IEEE-CIS',
      },
      {
        runId: 'mlf-run-041',
        name: 'xgb-smote-baseline-v1',
        modelType: 'XGBoost + SMOTE',
        createdAt: '2026-11-16 14:30 UTC',
        prAuc: 0.784,
        rocAuc: 0.964,
        f1AtThreshold: 0.738,
        precisionAt90Recall: 0.591,
        p99LatencyMs: 5.1,
        imbalanceStrategy: 'SMOTE (1:10 synthetic ratio)',
        blendWeightXgb: 1.0,
        blendWeightGnn: 0.0,
        status: 'archived',
        dataset: 'PaySim',
      },
    ];

    this.prCurve = [
      { recall: 0.1, xgbPrecision: 0.99, gnnPrecision: 0.98, blendedPrecision: 0.995 },
      { recall: 0.2, xgbPrecision: 0.978, gnnPrecision: 0.962, blendedPrecision: 0.988 },
      { recall: 0.3, xgbPrecision: 0.964, gnnPrecision: 0.941, blendedPrecision: 0.979 },
      { recall: 0.4, xgbPrecision: 0.946, gnnPrecision: 0.916, blendedPrecision: 0.968 },
      { recall: 0.5, xgbPrecision: 0.922, gnnPrecision: 0.884, blendedPrecision: 0.952 },
      { recall: 0.6, xgbPrecision: 0.889, gnnPrecision: 0.842, blendedPrecision: 0.928 },
      { recall: 0.7, xgbPrecision: 0.846, gnnPrecision: 0.791, blendedPrecision: 0.896 },
      { recall: 0.8, xgbPrecision: 0.792, gnnPrecision: 0.728, blendedPrecision: 0.851 },
      { recall: 0.85, xgbPrecision: 0.758, gnnPrecision: 0.688, blendedPrecision: 0.816 },
      { recall: 0.9, xgbPrecision: 0.712, gnnPrecision: 0.642, blendedPrecision: 0.768 },
      { recall: 0.95, xgbPrecision: 0.604, gnnPrecision: 0.515, blendedPrecision: 0.674 },
      { recall: 1.0, xgbPrecision: 0.087, gnnPrecision: 0.087, blendedPrecision: 0.087 },
    ];

    this.retrainingJobs = [
      {
        id: 'RT-011',
        triggeredAt: '2026-11-18 19:20 UTC',
        reason: 'Scheduled weekly Feast materialization + 142 analyst feedback labels',
        status: 'completed',
        progressPct: 100,
        feedbackSamplesUsed: 142,
        resultingRunId: 'mlf-run-048',
      },
      {
        id: 'RT-010',
        triggeredAt: '2026-11-15 08:14 UTC',
        reason: 'Evidently AI PSI breach on amount_vs_user_avg (PSI 0.24 > 0.20)',
        status: 'completed',
        progressPct: 100,
        feedbackSamplesUsed: 98,
        resultingRunId: 'mlf-run-045',
      },
    ];
  }

  private seedInitialHistory(): void {
    const now = Date.now();
    // Pre-populate 36 transactions including several genuine fraud/review cases
    for (let i = 35; i >= 0; i--) {
      const forceFraud = i === 29 || i === 21 || i === 14 || i === 7 || i === 2;
      const forceReview = i === 25 || i === 18 || i === 10 || i === 5;
      const ts = new Date(now - i * 4500);
      this.generateAndIngestTransaction({ forceFraud, forceReview, timestamp: ts, silent: true });
    }

    // Seed clean throughput & drift timeline points
    for (let i = 15; i >= 0; i--) {
      const t = new Date(now - i * 10000);
      this.throughputHistory.push({
        time: formatShortTime(t),
        tps: round(125 + Math.sin(i * 0.6) * 18 + Math.random() * 14, 0),
        flagged: Math.floor(1 + Math.random() * 4),
        avgLatencyMs: round(9.4 + Math.random() * 3.8, 1),
      });
      this.driftHistory.push({
        time: formatShortTime(t),
        datasetPsi: round(0.06 + Math.random() * 0.04, 3),
        driftedShare: 0,
        fraudRatePct: round(0.84 + Math.random() * 0.22, 2),
      });
    }

    // Mark a couple cases with analyst feedback so dashboard shows active workflow
    if (this.cases.length >= 3) {
      this.cases[this.cases.length - 1].status = 'confirmed_fraud';
      this.cases[this.cases.length - 1].feedbackLabel = 'fraud';
      this.cases[this.cases.length - 1].notes = 'Confirmed ATO via shared emulator device DEV-F991A.';
      this.cases[this.cases.length - 2].status = 'false_positive';
      this.cases[this.cases.length - 2].feedbackLabel = 'legitimate';
      this.cases[this.cases.length - 2].notes = 'Cardholder verified international flight booking via SMS.';
    }
  }

  private startLoop(): void {
    if (this.intervalHandle) {
      clearInterval(this.intervalHandle);
    }
    const intervalMs = Math.max(450, Math.round(1000 / this.streamControl.ratePerSec));
    this.intervalHandle = setInterval(() => {
      if (!this.streamControl.running) return;
      const tx = this.generateAndIngestTransaction({ silent: false });
      this.updateThroughputAndDriftHistory(tx);
    }, intervalMs);
  }

  private updateThroughputAndDriftHistory(tx: ScoredTransaction): void {
    const label = formatShortTime(new Date());
    const baseTps = Math.round(90 * this.streamControl.ratePerSec + Math.random() * 35);
    this.throughputHistory.push({
      time: label,
      tps: baseTps,
      flagged: tx.decision === 'APPROVE' ? 0 : 1,
      avgLatencyMs: tx.latencyMs,
    });
    if (this.throughputHistory.length > 24) {
      this.throughputHistory.shift();
    }

    const drift = this.computeDriftReport(false);
    this.driftHistory.push({
      time: label,
      datasetPsi: drift.datasetPsi,
      driftedShare: round(drift.driftedShare * 100, 0),
      fraudRatePct: this.getMetrics().fraudRatePct,
    });
    if (this.driftHistory.length > 24) {
      this.driftHistory.shift();
    }

    // Auto-retrain trigger check when drift breaches threshold
    if (
      this.autoRetrainEnabled &&
      drift.driftAlertActive &&
      Date.now() - this.lastRetrainTriggerMs > 45000 &&
      !this.retrainingJobs.some((j) => j.status === 'running')
    ) {
      this.lastRetrainTriggerMs = Date.now();
      this.triggerRetraining(
        `Evidently AI auto-trigger: Dataset PSI ${drift.datasetPsi.toFixed(2)} exceeded ${this.driftThresholdPsi.toFixed(2)} threshold`
      );
    }
  }

  private generateAndIngestTransaction(opts: {
    forceFraud?: boolean;
    forceReview?: boolean;
    ringIndex?: number;
    timestamp?: Date;
    silent?: boolean;
  }): ScoredTransaction {
    const isFraud =
      opts.forceFraud ??
      (this.streamControl.driftInjected ? Math.random() < 0.22 : Math.random() < 0.085);
    const isReview = !isFraud && (opts.forceReview ?? Math.random() < 0.09);

    let account: AccountBaseline;
    let deviceId: string;
    let cardHash: string;
    let ipAddress: string;
    let amount: number;
    let merchant = randomPick(MERCHANTS);
    let location: string;
    let txCount1h: number;
    let geoDistanceKm: number;
    let deviceChangeFlag: 0 | 1;

    if (isFraud) {
      const ring =
        opts.ringIndex !== undefined
          ? this.rings[opts.ringIndex % this.rings.length]
          : randomPick(this.rings);
      const ringAccounts = ring.nodes.filter((n) => n.type === 'account');
      const pickedNode = randomPick(ringAccounts);
      account = this.accounts.get(pickedNode.id)!;
      const devNode = ring.nodes.find((n) => n.type === 'device')!;
      const cardNode = ring.nodes.find((n) => n.type === 'card')!;
      const ipNode = ring.nodes.find((n) => n.type === 'ip')!;

      deviceId = devNode.id;
      cardHash = cardNode.id;
      ipAddress = ipNode.id;
      merchant = randomPick(MERCHANTS.filter((m) => m.risk >= 0.45));
      amount = round(account.avgAmount * (4.2 + Math.random() * 5.8), 2);
      location = randomPick(CITIES.slice(6));
      txCount1h = 5 + Math.floor(Math.random() * 8);
      geoDistanceKm = Math.round(1400 + Math.random() * 6200);
      deviceChangeFlag = 1;
    } else if (isReview) {
      const normalAccounts = Array.from(this.accounts.values()).filter((a) => !a.ringId);
      account = randomPick(normalAccounts);
      deviceId = Math.random() < 0.5 ? account.primaryDevice : `DEV-NEW-${Math.floor(100 + Math.random() * 899)}`;
      cardHash = account.primaryCard;
      ipAddress = `172.58.${Math.floor(10 + Math.random() * 200)}.${Math.floor(10 + Math.random() * 230)}`;
      merchant = randomPick(MERCHANTS.filter((m) => m.risk >= 0.24 && m.risk <= 0.68));
      amount = round(account.avgAmount * (2.3 + Math.random() * 2.1), 2);
      location = Math.random() < 0.5 ? account.homeCity : randomPick(CITIES.slice(4, 8));
      txCount1h = 3 + Math.floor(Math.random() * 3);
      geoDistanceKm = location === account.homeCity ? Math.round(Math.random() * 40) : Math.round(600 + Math.random() * 1900);
      deviceChangeFlag = deviceId === account.primaryDevice ? 0 : 1;
    } else {
      const normalAccounts = Array.from(this.accounts.values()).filter((a) => !a.ringId);
      account = randomPick(normalAccounts);
      deviceId = account.primaryDevice;
      cardHash = account.primaryCard;
      ipAddress = `104.28.${Math.floor(10 + Math.random() * 200)}.${Math.floor(10 + Math.random() * 230)}`;
      merchant = randomPick(MERCHANTS.filter((m) => m.risk <= 0.48));
      // When drift is injected, even normal traffic shifts upward in amount ratio and velocity
      const driftMult = this.streamControl.driftInjected ? 2.4 : 1.0;
      amount = round(account.avgAmount * (0.45 + Math.random() * 1.1) * driftMult, 2);
      location = this.streamControl.driftInjected && Math.random() < 0.45 ? randomPick(CITIES.slice(5)) : account.homeCity;
      txCount1h = this.streamControl.driftInjected
        ? 3 + Math.floor(Math.random() * 4)
        : 1 + Math.floor(Math.random() * 2);
      geoDistanceKm =
        location === account.homeCity
          ? Math.round(Math.random() * 22)
          : Math.round(900 + Math.random() * 2800);
      deviceChangeFlag = this.streamControl.driftInjected && Math.random() < 0.38 ? 1 : 0;
    }

    const txType =
      this.streamControl.dataset === 'PaySim' ? randomPick(PAYSIM_TYPES) : randomPick(IEEE_TYPES);

    return this.scorePayload(
      {
        accountId: account.accountId,
        amount,
        merchantName: merchant.name,
        merchantCategory: merchant.category,
        location,
        deviceId,
        cardHash,
        ipAddress,
        txType,
        txCount1h,
        geoDistanceKm,
        deviceChangeFlag,
        ingestIntoStream: true,
      },
      opts.timestamp ?? new Date(),
      opts.silent ?? false
    );
  }

  public scorePayload(
    req: ScoreTransactionRequest,
    timestamp = new Date(),
    silent = false
  ): ScoredTransaction {
    const startNs = performance.now();

    const account = this.accounts.get(req.accountId);
    const avgAmount = account ? account.avgAmount : 185;
    const accountAgeDays = account ? account.accountAgeDays : 14;
    const merchantMeta = MERCHANTS.find((m) => m.name === req.merchantName);
    const merchantRiskIndex = merchantMeta ? merchantMeta.risk : 0.45;

    // Feast online feature retrieval simulation
    const feastLookupMs = round(1.8 + Math.random() * 2.4, 2);
    const txCount1h = req.txCount1h ?? (account ? Math.min(12, account.recentTimestamps.length + 1) : 2);
    const txCount24h = txCount1h + Math.floor(2 + Math.random() * 7);
    const amountVsUserAvg = round(req.amount / Math.max(10, avgAmount), 2);
    const deviceChangeFlag: 0 | 1 =
      req.deviceChangeFlag !== undefined
        ? req.deviceChangeFlag
        : account && req.deviceId !== account.primaryDevice
          ? 1
          : 0;
    const geoDistanceKm = req.geoDistanceKm ?? (account && req.location !== account.homeCity ? 1850 : 12);

    const sharedByDevice = this.deviceToAccounts.get(req.deviceId)?.size ?? 1;
    const sharedByCard = this.cardToAccounts.get(req.cardHash)?.size ?? 1;
    const cardSharedAccounts = Math.max(sharedByDevice, sharedByCard);

    const features: FeastFeatureVector = {
      txCount1h,
      txCount24h,
      amountVsUserAvg,
      deviceChangeFlag,
      geoDistanceKm,
      cardSharedAccounts,
      merchantRiskIndex,
      accountAgeDays,
      feastLookupMs,
    };

    // XGBoost additive log-odds scoring + per-feature attribution
    const baseLogOdds = -4.25;
    const cAmount = clamp((amountVsUserAvg - 1.4) * 0.58, -0.35, 2.35);
    const cVelocity = clamp((txCount1h - 1.5) * 0.46, -0.25, 2.1);
    const cGeo = clamp((geoDistanceKm / 1500) * 0.62, 0, 1.45);
    const cDevice = deviceChangeFlag === 1 ? 1.15 : -0.35;
    const cMerchant = (merchantRiskIndex - 0.25) * 1.9;
    const cAge = accountAgeDays < 30 ? 0.72 : -0.28;

    const xgbLogOdds = baseLogOdds + cAmount + cVelocity + cGeo + cDevice + cMerchant + cAge;
    const xgbScore = round(clamp(sigmoid(xgbLogOdds), 0.01, 0.99), 3);

    // PyTorch Geometric GNN score: neighborhood ring aggregation
    const matchedRing =
      this.rings.find(
        (r) =>
          r.id === account?.ringId ||
          r.nodes.some((n) => n.id === req.deviceId || n.id === req.cardHash || n.id === req.ipAddress)
      ) ?? null;

    let gnnScore: number;
    if (matchedRing) {
      gnnScore = round(clamp(matchedRing.gnnConfidence - 0.04 + Math.random() * 0.07, 0.74, 0.98), 3);
    } else if (cardSharedAccounts >= 2) {
      gnnScore = round(clamp(0.42 + cardSharedAccounts * 0.14, 0.45, 0.85), 3);
    } else {
      gnnScore = round(clamp(0.04 + deviceChangeFlag * 0.14 + Math.random() * 0.08, 0.02, 0.34), 3);
    }

    const blendedScore = round(
      clamp(this.modelConfig.xgbWeight * xgbScore + this.modelConfig.gnnWeight * gnnScore, 0.01, 0.99),
      3
    );

    const decision: DecisionAction =
      blendedScore >= this.modelConfig.blockThreshold
        ? 'BLOCK'
        : blendedScore >= this.modelConfig.reviewThreshold
          ? 'REVIEW'
          : 'APPROVE';

    const attributions: FeatureAttribution[] = [
      {
        feature: 'amount_vs_user_avg',
        label: 'Amount vs. User Avg',
        value: `${amountVsUserAvg.toFixed(2)}x`,
        contribution: round(cAmount, 2),
      },
      {
        feature: 'tx_count_1h',
        label: '1h Transaction Velocity',
        value: `${txCount1h} tx/hr`,
        contribution: round(cVelocity, 2),
      },
      {
        feature: 'gnn_ring_embedding',
        label: 'PyG Ring Neighborhood',
        value: matchedRing ? matchedRing.id : `${cardSharedAccounts} linked`,
        contribution: round((gnnScore - 0.15) * 2.1, 2),
      },
      {
        feature: 'device_change_flag',
        label: 'Device Fingerprint Shift',
        value: deviceChangeFlag === 1 ? 'New Device' : 'Recognized',
        contribution: round(cDevice, 2),
      },
      {
        feature: 'geo_distance_km',
        label: 'Geo Distance Delta',
        value: `${geoDistanceKm.toLocaleString()} km`,
        contribution: round(cGeo, 2),
      },
      {
        feature: 'merchant_risk_index',
        label: 'Merchant Category Risk',
        value: `${Math.round(merchantRiskIndex * 100)}/100`,
        contribution: round(cMerchant, 2),
      },
    ].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

    const computeMs = performance.now() - startNs;
    const latencyMs = round(feastLookupMs + 4.1 + computeMs + Math.random() * 3.2, 1);

    this.txCounter += 1;
    const txId = `TX-${this.txCounter}`;
    let caseId: string | null = null;

    if (req.ingestIntoStream !== false) {
      this.totalProcessed += 1;
      this.latencies.push(latencyMs);
      if (this.latencies.length > 120) this.latencies.shift();

      if (decision !== 'APPROVE') {
        this.flaggedCount += 1;
        if (decision === 'BLOCK') {
          this.blockedCount += 1;
          this.preventedLossUsd = round(this.preventedLossUsd + req.amount, 2);
        }
        this.caseCounter += 1;
        caseId = `CASE-${this.caseCounter}`;
        const topReason = `${attributions[0].label} (${attributions[0].value})`;

        const newCase: AnalystCase = {
          id: caseId,
          transactionId: txId,
          createdAt: timestamp.toISOString(),
          updatedAt: timestamp.toISOString(),
          accountId: req.accountId,
          merchantName: req.merchantName,
          amount: req.amount,
          blendedScore,
          xgbScore,
          gnnScore,
          decision,
          ringId: matchedRing?.id ?? null,
          status: 'open',
          feedbackLabel: 'unlabeled',
          assignee: 'Elena Lin',
          notes: '',
          topReason,
        };
        this.cases.unshift(newCase);
        if (this.cases.length > 60) this.cases.pop();

        if (matchedRing) {
          if (!matchedRing.linkedCaseIds.includes(caseId)) {
            matchedRing.linkedCaseIds.unshift(caseId);
            matchedRing.totalExposure = round(matchedRing.totalExposure + req.amount, 2);
          }
        }
      }
    }

    const scored: ScoredTransaction = {
      id: txId,
      timestamp: timestamp.toISOString(),
      dataset: this.streamControl.dataset,
      accountId: req.accountId,
      cardHash: req.cardHash,
      deviceId: req.deviceId,
      ipAddress: req.ipAddress,
      merchantName: req.merchantName,
      merchantCategory: req.merchantCategory,
      location: req.location,
      amount: req.amount,
      currency: 'USD',
      txType: req.txType,
      xgbScore,
      gnnScore,
      blendedScore,
      decision,
      latencyMs,
      ringId: matchedRing?.id ?? null,
      features,
      attributions,
      caseId,
    };

    if (req.ingestIntoStream !== false) {
      this.transactions.unshift(scored);
      if (this.transactions.length > 80) {
        this.transactions.pop();
      }
      if (!silent) {
        this.emit('transaction', {
          transaction: scored,
          metrics: this.getMetrics(),
          driftReport: this.computeDriftReport(false),
          throughputPoint: this.throughputHistory[this.throughputHistory.length - 1],
          newCase: caseId ? this.cases.find((c) => c.id === caseId) : null,
        });
      }
    }

    return scored;
  }

  public injectFraudBurst(): ScoredTransaction[] {
    const burst: ScoredTransaction[] = [];
    for (let i = 0; i < 4; i++) {
      const tx = this.generateAndIngestTransaction({
        forceFraud: true,
        ringIndex: i % this.rings.length,
        silent: false,
      });
      burst.push(tx);
    }
    this.emit('state_sync', this.getSnapshot());
    return burst;
  }

  public updateStreamControl(patch: Partial<StreamControlState>): StreamControlState {
    const rateChanged = patch.ratePerSec !== undefined && patch.ratePerSec !== this.streamControl.ratePerSec;
    this.streamControl = {
      ...this.streamControl,
      ...patch,
    };
    if (rateChanged) {
      this.startLoop();
    }
    // If drift injection toggled on, immediately push 10 shifted transactions so PSI reacts right away
    if (patch.driftInjected === true) {
      for (let i = 0; i < 10; i++) {
        this.generateAndIngestTransaction({ silent: true });
      }
      this.updateThroughputAndDriftHistory(this.transactions[0]);
    }
    this.emit('state_sync', this.getSnapshot());
    return this.streamControl;
  }

  public updateCase(
    caseId: string,
    patch: { status?: CaseStatus; feedbackLabel?: FeedbackLabel; notes?: string }
  ): AnalystCase | null {
    const found = this.cases.find((c) => c.id === caseId);
    if (!found) return null;

    if (patch.status) found.status = patch.status;
    if (patch.feedbackLabel) found.feedbackLabel = patch.feedbackLabel;
    if (patch.notes !== undefined) found.notes = patch.notes;

    // Keep status and feedback label synchronized automatically
    if (patch.status === 'confirmed_fraud' && !patch.feedbackLabel) {
      found.feedbackLabel = 'fraud';
    } else if (patch.status === 'false_positive' && !patch.feedbackLabel) {
      found.feedbackLabel = 'legitimate';
    }
    found.updatedAt = new Date().toISOString();

    this.emit('state_sync', this.getSnapshot());
    return found;
  }

  public performRingAction(ringId: string, action: 'freeze' | 'escalate' | 'release'): FraudRing | null {
    const ring = this.rings.find((r) => r.id === ringId);
    if (!ring) return null;

    if (action === 'freeze') {
      ring.status = 'frozen';
      ring.nodes.forEach((n) => {
        n.frozen = true;
      });
      this.cases
        .filter((c) => c.ringId === ringId && c.status === 'open')
        .forEach((c) => {
          c.status = 'confirmed_fraud';
          c.feedbackLabel = 'fraud';
          c.notes = `Auto-confirmed via ${ringId} ring freeze action.`;
          c.updatedAt = new Date().toISOString();
        });
    } else if (action === 'escalate') {
      ring.status = 'escalated';
      this.cases
        .filter((c) => c.ringId === ringId && c.status === 'open')
        .forEach((c) => {
          c.status = 'investigating';
          c.updatedAt = new Date().toISOString();
        });
    } else if (action === 'release') {
      ring.status = 'active';
      ring.nodes.forEach((n) => {
        n.frozen = false;
      });
    }

    this.emit('state_sync', this.getSnapshot());
    return ring;
  }

  public updateModelConfig(patch: Partial<ModelConfig>): ModelConfig {
    if (patch.activeRunId) {
      const run = this.mlflowRuns.find((r) => r.runId === patch.activeRunId);
      if (run) {
        this.mlflowRuns.forEach((r) => {
          if (r.status === 'production') r.status = 'staging';
        });
        run.status = 'production';
        this.modelConfig.activeRunId = run.runId;
        this.modelConfig.xgbWeight = run.blendWeightXgb;
        this.modelConfig.gnnWeight = run.blendWeightGnn;
      }
    }
    if (patch.xgbWeight !== undefined) {
      const w = round(clamp(patch.xgbWeight, 0, 1), 2);
      this.modelConfig.xgbWeight = w;
      this.modelConfig.gnnWeight = round(1 - w, 2);
    }
    if (patch.reviewThreshold !== undefined) {
      this.modelConfig.reviewThreshold = round(clamp(patch.reviewThreshold, 0.25, 0.8), 2);
    }
    if (patch.blockThreshold !== undefined) {
      this.modelConfig.blockThreshold = round(
        clamp(patch.blockThreshold, this.modelConfig.reviewThreshold + 0.05, 0.96),
        2
      );
    }

    this.emit('state_sync', this.getSnapshot());
    return this.modelConfig;
  }

  public setDriftSettings(patch: { autoRetrainEnabled?: boolean; driftThresholdPsi?: number }): DriftReport {
    if (patch.autoRetrainEnabled !== undefined) {
      this.autoRetrainEnabled = patch.autoRetrainEnabled;
    }
    if (patch.driftThresholdPsi !== undefined) {
      this.driftThresholdPsi = round(clamp(patch.driftThresholdPsi, 0.1, 0.4), 2);
    }
    const report = this.computeDriftReport(false);
    this.emit('state_sync', this.getSnapshot());
    return report;
  }

  public triggerRetraining(reason = 'Manual analyst retrain trigger'): RetrainingJob {
    this.jobCounter += 1;
    const labeledCount = this.cases.filter((c) => c.feedbackLabel !== 'unlabeled').length + 142;
    const job: RetrainingJob = {
      id: `RT-0${this.jobCounter}`,
      triggeredAt: new Date().toISOString().replace('T', ' ').slice(0, 16) + ' UTC',
      reason,
      status: 'running',
      progressPct: 15,
      feedbackSamplesUsed: labeledCount,
      resultingRunId: null,
    };
    this.retrainingJobs.unshift(job);
    if (this.retrainingJobs.length > 8) this.retrainingJobs.pop();
    this.emit('state_sync', this.getSnapshot());

    // Simulate realistic multi-stage Kubernetes training job completion
    const steps = [42, 74, 100];
    steps.forEach((pct, idx) => {
      setTimeout(() => {
        job.progressPct = pct;
        if (pct === 100) {
          job.status = 'completed';
          const nextNum = 48 + this.jobCounter - 12;
          const runId = `mlf-run-0${nextNum}`;
          job.resultingRunId = runId;

          const currentProd = this.mlflowRuns.find((r) => r.status === 'production');
          const basePrAuc = currentProd ? currentProd.prAuc : 0.894;
          const improvedPrAuc = round(clamp(basePrAuc + 0.008 + Math.random() * 0.007, 0.85, 0.945), 3);

          this.mlflowRuns.forEach((r) => {
            if (r.status === 'production') r.status = 'staging';
          });
          this.mlflowRuns.unshift({
            runId,
            name: `ensemble-retrained-r${nextNum}`,
            modelType: 'Blended Ensemble',
            createdAt: new Date().toISOString().replace('T', ' ').slice(0, 16) + ' UTC',
            prAuc: improvedPrAuc,
            rocAuc: round(clamp(improvedPrAuc + 0.09, 0.95, 0.994), 3),
            f1AtThreshold: round(improvedPrAuc - 0.048, 3),
            precisionAt90Recall: round(improvedPrAuc - 0.118, 3),
            p99LatencyMs: round(13.6 + Math.random() * 1.4, 1),
            imbalanceStrategy: `scale_pos_weight=110 + ${labeledCount} analyst labels`,
            blendWeightXgb: this.modelConfig.xgbWeight,
            blendWeightGnn: this.modelConfig.gnnWeight,
            status: 'production',
            dataset: this.streamControl.dataset,
          });
          this.modelConfig.activeRunId = runId;

          // Clear injected drift after fresh baseline re-calibration
          if (this.streamControl.driftInjected) {
            this.streamControl.driftInjected = false;
            for (let k = 0; k < 14; k++) {
              this.generateAndIngestTransaction({ forceFraud: false, forceReview: false, silent: true });
            }
          }
        }
        this.emit('state_sync', this.getSnapshot());
      }, (idx + 1) * 1400);
    });

    return job;
  }

  private computePsi(refPcts: number[], curPcts: number[]): number {
    let psi = 0;
    for (let i = 0; i < refPcts.length; i++) {
      const r = Math.max(0.005, refPcts[i] / 100);
      const c = Math.max(0.005, curPcts[i] / 100);
      psi += (c - r) * Math.log(c / r);
    }
    return round(Math.abs(psi), 3);
  }

  private computeDriftReport(includeEmptyCheck = false): DriftReport {
    const window = this.transactions.slice(0, 45);
    const n = Math.max(1, window.length);

    // 1. amount_vs_user_avg
    const refAmountBins = [46, 32, 14, 6, 2];
    const amtCounts = [0, 0, 0, 0, 0];
    let amtSum = 0;
    for (const t of window) {
      const v = t.features.amountVsUserAvg;
      amtSum += v;
      if (v < 1.0) amtCounts[0]++;
      else if (v < 2.0) amtCounts[1]++;
      else if (v < 3.5) amtCounts[2]++;
      else if (v < 5.5) amtCounts[3]++;
      else amtCounts[4]++;
    }
    const curAmountBins = amtCounts.map((c) => round((c / n) * 100, 1));
    const amtPsi = this.computePsi(refAmountBins, curAmountBins);

    // 2. tx_count_1h
    const refVelBins = [58, 26, 10, 4, 2];
    const velCounts = [0, 0, 0, 0, 0];
    let velSum = 0;
    for (const t of window) {
      const v = t.features.txCount1h;
      velSum += v;
      if (v <= 1) velCounts[0]++;
      else if (v <= 2) velCounts[1]++;
      else if (v <= 4) velCounts[2]++;
      else if (v <= 6) velCounts[3]++;
      else velCounts[4]++;
    }
    const curVelBins = velCounts.map((c) => round((c / n) * 100, 1));
    const velPsi = this.computePsi(refVelBins, curVelBins);

    // 3. geo_distance_km
    const refGeoBins = [74, 14, 7, 3, 2];
    const geoCounts = [0, 0, 0, 0, 0];
    let geoSum = 0;
    for (const t of window) {
      const v = t.features.geoDistanceKm;
      geoSum += v;
      if (v < 50) geoCounts[0]++;
      else if (v < 500) geoCounts[1]++;
      else if (v < 1500) geoCounts[2]++;
      else if (v < 3500) geoCounts[3]++;
      else geoCounts[4]++;
    }
    const curGeoBins = geoCounts.map((c) => round((c / n) * 100, 1));
    const geoPsi = this.computePsi(refGeoBins, curGeoBins);

    // 4. device_change_flag
    const refDevBins = [88, 12];
    const devNewCount = window.filter((t) => t.features.deviceChangeFlag === 1).length;
    const curDevNewPct = round((devNewCount / n) * 100, 1);
    const curDevBins = [round(100 - curDevNewPct, 1), curDevNewPct];
    const devPsi = this.computePsi(refDevBins, curDevBins);

    // 5. blended_score prediction drift
    const refScoreBins = [84, 9, 4, 2, 1];
    const scoreCounts = [0, 0, 0, 0, 0];
    let scoreSum = 0;
    for (const t of window) {
      const s = t.blendedScore;
      scoreSum += s;
      if (s < 0.2) scoreCounts[0]++;
      else if (s < 0.45) scoreCounts[1]++;
      else if (s < 0.65) scoreCounts[2]++;
      else if (s < 0.82) scoreCounts[3]++;
      else scoreCounts[4]++;
    }
    const curScoreBins = scoreCounts.map((c) => round((c / n) * 100, 1));
    const scorePsi = this.computePsi(refScoreBins, curScoreBins);

    const makeFeatureMetric = (
      feature: string,
      displayName: string,
      psi: number,
      refMean: number,
      curMean: number,
      labels: string[],
      refBins: number[],
      curBins: number[]
    ): FeatureDriftMetric => {
      const ksStat = round(clamp(psi * 0.68, 0.02, 0.72), 3);
      const pValue = round(clamp(Math.exp(-psi * 16), 0.0001, 0.94), 4);
      return {
        feature,
        displayName,
        psi,
        ksStat,
        pValue,
        referenceMean: refMean,
        currentMean: round(curMean, 2),
        drifted: psi >= this.driftThresholdPsi,
        bins: labels.map((bucket, idx) => ({
          bucket,
          referencePct: refBins[idx],
          currentPct: curBins[idx],
        })),
      };
    };

    const features: FeatureDriftMetric[] = [
      makeFeatureMetric(
        'amount_vs_user_avg',
        'Amount vs. User Average',
        amtPsi,
        1.12,
        amtSum / n,
        ['<1x', '1-2x', '2-3.5x', '3.5-5.5x', '>5.5x'],
        refAmountBins,
        curAmountBins
      ),
      makeFeatureMetric(
        'tx_count_1h',
        '1h Transaction Velocity',
        velPsi,
        1.48,
        velSum / n,
        ['1', '2', '3-4', '5-6', '7+'],
        refVelBins,
        curVelBins
      ),
      makeFeatureMetric(
        'geo_distance_km',
        'Geolocation Delta (km)',
        geoPsi,
        195,
        geoSum / n,
        ['<50', '50-500', '0.5-1.5k', '1.5-3.5k', '>3.5k'],
        refGeoBins,
        curGeoBins
      ),
      makeFeatureMetric(
        'device_change_flag',
        'Device Change Rate',
        devPsi,
        0.12,
        devNewCount / n,
        ['Known Device', 'New Device'],
        refDevBins,
        curDevBins
      ),
      makeFeatureMetric(
        'blended_score',
        'Blended Risk Score Output',
        scorePsi,
        0.11,
        scoreSum / n,
        ['0-0.2', '0.2-0.45', '0.45-0.65', '0.65-0.82', '0.82+'],
        refScoreBins,
        curScoreBins
      ),
    ];

    const driftedFeatureCount = features.filter((f) => f.drifted).length;
    const datasetPsi = round(features.reduce((acc, f) => acc + f.psi, 0) / features.length, 3);
    const driftedShare = round(driftedFeatureCount / features.length, 2);

    return {
      evaluatedAt: new Date().toISOString(),
      windowSize: includeEmptyCheck ? 0 : window.length,
      driftedFeatureCount,
      totalFeatures: features.length,
      driftedShare,
      datasetPsi,
      driftThresholdPsi: this.driftThresholdPsi,
      autoRetrainEnabled: this.autoRetrainEnabled,
      driftAlertActive: datasetPsi >= this.driftThresholdPsi || driftedShare >= 0.4,
      driftInjected: this.streamControl.driftInjected,
      features,
      history: [...this.driftHistory],
      jobs: [...this.retrainingJobs],
    };
  }

  public getMetrics(): PlatformMetrics {
    const sortedLat = [...this.latencies].sort((a, b) => a - b);
    const avgLatencyMs =
      sortedLat.length > 0
        ? round(sortedLat.reduce((a, b) => a + b, 0) / sortedLat.length, 1)
        : 10.4;
    const p99LatencyMs =
      sortedLat.length > 0 ? sortedLat[Math.min(sortedLat.length - 1, Math.floor(sortedLat.length * 0.99))] : 14.2;

    return {
      totalProcessed: this.totalProcessed,
      flaggedCount: this.flaggedCount,
      blockedCount: this.blockedCount,
      fraudRatePct: round((this.blockedCount / Math.max(1, this.totalProcessed)) * 100, 2),
      avgLatencyMs,
      p99LatencyMs,
      openCases: this.cases.filter((c) => c.status === 'open' || c.status === 'investigating').length,
      activeRings: this.rings.filter((r) => r.status !== 'frozen').length,
      feedbackLabelsCollected: this.cases.filter((c) => c.feedbackLabel !== 'unlabeled').length + 142,
      preventedLossUsd: this.preventedLossUsd,
    };
  }

  private getEdaSummaries(): Record<DatasetProfile, DatasetEdaSummary> {
    return {
      'IEEE-CIS': {
        profile: 'IEEE-CIS',
        totalRows: '590,540',
        fraudRatioPct: 0.87,
        legitimateCount: '585,402',
        fraudCount: '5,138',
        imbalanceRatio: '1 : 114',
        feastFeatureCount: 28,
        keySignals: [
          { feature: 'amount_vs_user_avg', fraudMedian: '4.82x', legitMedian: '0.96x', lift: '5.0x' },
          { feature: 'tx_count_1h', fraudMedian: '6 tx/hr', legitMedian: '1 tx/hr', lift: '6.0x' },
          { feature: 'device_change_flag', fraudMedian: '78.4%', legitMedian: '9.2%', lift: '8.5x' },
          { feature: 'card_shared_accounts', fraudMedian: '3.4 accts', legitMedian: '1.0 accts', lift: '3.4x' },
        ],
      },
      PaySim: {
        profile: 'PaySim',
        totalRows: '6,362,620',
        fraudRatioPct: 0.13,
        legitimateCount: '6,354,407',
        fraudCount: '8,213',
        imbalanceRatio: '1 : 774',
        feastFeatureCount: 22,
        keySignals: [
          { feature: 'balance_drain_ratio', fraudMedian: '0.99', legitMedian: '0.14', lift: '7.1x' },
          { feature: 'tx_count_1h', fraudMedian: '5 tx/hr', legitMedian: '1 tx/hr', lift: '5.0x' },
          { feature: 'cash_out_hop_degree', fraudMedian: '4.1 hops', legitMedian: '1.1 hops', lift: '3.7x' },
          { feature: 'account_age_days', fraudMedian: '6 days', legitMedian: '248 days', lift: '41.3x' },
        ],
      },
    };
  }

  public getSnapshot(): PlatformStateSnapshot {
    return {
      metrics: this.getMetrics(),
      stream: { ...this.streamControl },
      modelConfig: { ...this.modelConfig },
      transactions: [...this.transactions],
      cases: [...this.cases],
      rings: this.rings.map((r) => ({ ...r, nodes: [...r.nodes], edges: [...r.edges] })),
      mlflowRuns: [...this.mlflowRuns],
      prCurve: [...this.prCurve],
      driftReport: this.computeDriftReport(false),
      throughputHistory: [...this.throughputHistory],
      edaSummaries: this.getEdaSummaries(),
    };
  }
}

export const fraudPlatform = new FraudPlatformEngine();
