export type DatasetProfile = 'IEEE-CIS' | 'PaySim';

export type DecisionAction = 'APPROVE' | 'REVIEW' | 'BLOCK';

export type CaseStatus = 'open' | 'investigating' | 'confirmed_fraud' | 'false_positive';

export type FeedbackLabel = 'unlabeled' | 'fraud' | 'legitimate';

export interface FeastFeatureVector {
  txCount1h: number;
  txCount24h: number;
  amountVsUserAvg: number;
  deviceChangeFlag: 0 | 1;
  geoDistanceKm: number;
  cardSharedAccounts: number;
  merchantRiskIndex: number;
  accountAgeDays: number;
  feastLookupMs: number;
}

export interface FeatureAttribution {
  feature: string;
  label: string;
  value: string;
  contribution: number;
}

export interface ScoredTransaction {
  id: string;
  timestamp: string;
  dataset: DatasetProfile;
  accountId: string;
  cardHash: string;
  deviceId: string;
  ipAddress: string;
  merchantName: string;
  merchantCategory: string;
  location: string;
  amount: number;
  currency: string;
  txType: string;
  xgbScore: number;
  gnnScore: number;
  blendedScore: number;
  decision: DecisionAction;
  latencyMs: number;
  ringId: string | null;
  features: FeastFeatureVector;
  attributions: FeatureAttribution[];
  caseId: string | null;
}

export interface AnalystCase {
  id: string;
  transactionId: string;
  createdAt: string;
  updatedAt: string;
  accountId: string;
  merchantName: string;
  amount: number;
  blendedScore: number;
  xgbScore: number;
  gnnScore: number;
  decision: DecisionAction;
  ringId: string | null;
  status: CaseStatus;
  feedbackLabel: FeedbackLabel;
  assignee: string;
  notes: string;
  topReason: string;
}

export type GraphEntityType = 'account' | 'device' | 'card' | 'ip';

export interface GraphNode {
  id: string;
  label: string;
  type: GraphEntityType;
  riskScore: number;
  frozen: boolean;
  ringId: string;
  txCount: number;
  totalAmount: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: 'uses_device' | 'owns_card' | 'originates_ip' | 'transfers_to';
  weight: number;
}

export interface FraudRing {
  id: string;
  name: string;
  detectedAt: string;
  gnnConfidence: number;
  accountCount: number;
  sharedEntities: number;
  totalExposure: number;
  status: 'active' | 'frozen' | 'escalated';
  pattern: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  linkedCaseIds: string[];
}

export interface PrCurvePoint {
  recall: number;
  xgbPrecision: number;
  gnnPrecision: number;
  blendedPrecision: number;
}

export interface MlflowRun {
  runId: string;
  name: string;
  modelType: 'XGBoost + SMOTE' | 'XGBoost (Class Weights)' | 'PyG GraphSAGE' | 'Blended Ensemble';
  createdAt: string;
  prAuc: number;
  rocAuc: number;
  f1AtThreshold: number;
  precisionAt90Recall: number;
  p99LatencyMs: number;
  imbalanceStrategy: string;
  blendWeightXgb: number;
  blendWeightGnn: number;
  status: 'production' | 'staging' | 'archived';
  dataset: DatasetProfile;
}

export interface FeatureDriftMetric {
  feature: string;
  displayName: string;
  psi: number;
  ksStat: number;
  pValue: number;
  referenceMean: number;
  currentMean: number;
  drifted: boolean;
  bins: Array<{
    bucket: string;
    referencePct: number;
    currentPct: number;
  }>;
}

export interface DriftHistoryPoint {
  time: string;
  datasetPsi: number;
  driftedShare: number;
  fraudRatePct: number;
}

export interface RetrainingJob {
  id: string;
  triggeredAt: string;
  reason: string;
  status: 'completed' | 'running';
  progressPct: number;
  feedbackSamplesUsed: number;
  resultingRunId: string | null;
}

export interface DriftReport {
  evaluatedAt: string;
  windowSize: number;
  driftedFeatureCount: number;
  totalFeatures: number;
  driftedShare: number;
  datasetPsi: number;
  driftThresholdPsi: number;
  autoRetrainEnabled: boolean;
  driftAlertActive: boolean;
  driftInjected: boolean;
  features: FeatureDriftMetric[];
  history: DriftHistoryPoint[];
  jobs: RetrainingJob[];
}

export interface ModelConfig {
  activeRunId: string;
  xgbWeight: number;
  gnnWeight: number;
  reviewThreshold: number;
  blockThreshold: number;
}

export interface ThroughputPoint {
  time: string;
  tps: number;
  flagged: number;
  avgLatencyMs: number;
}

export interface StreamControlState {
  running: boolean;
  ratePerSec: number;
  dataset: DatasetProfile;
  driftInjected: boolean;
}

export interface PlatformMetrics {
  totalProcessed: number;
  flaggedCount: number;
  blockedCount: number;
  fraudRatePct: number;
  avgLatencyMs: number;
  p99LatencyMs: number;
  openCases: number;
  activeRings: number;
  feedbackLabelsCollected: number;
  preventedLossUsd: number;
}

export interface DatasetEdaSummary {
  profile: DatasetProfile;
  totalRows: string;
  fraudRatioPct: number;
  legitimateCount: string;
  fraudCount: string;
  imbalanceRatio: string;
  feastFeatureCount: number;
  keySignals: Array<{
    feature: string;
    fraudMedian: string;
    legitMedian: string;
    lift: string;
  }>;
}

export interface PlatformStateSnapshot {
  metrics: PlatformMetrics;
  stream: StreamControlState;
  modelConfig: ModelConfig;
  transactions: ScoredTransaction[];
  cases: AnalystCase[];
  rings: FraudRing[];
  mlflowRuns: MlflowRun[];
  prCurve: PrCurvePoint[];
  driftReport: DriftReport;
  throughputHistory: ThroughputPoint[];
  edaSummaries: Record<DatasetProfile, DatasetEdaSummary>;
}

export interface ScoreTransactionRequest {
  accountId: string;
  amount: number;
  merchantName: string;
  merchantCategory: string;
  location: string;
  deviceId: string;
  cardHash: string;
  ipAddress: string;
  txType: string;
  txCount1h?: number;
  geoDistanceKm?: number;
  deviceChangeFlag?: 0 | 1;
  ingestIntoStream?: boolean;
}
