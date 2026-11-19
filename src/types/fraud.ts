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

// Phase 2: Real FastAPI XGBoost Inference Types (via Express /api/v1/ml/* proxy)
export type MlDecision = 'Approved' | 'Review' | 'Blocked';

export interface MlPredictRequest {
  transaction_id: string;
  transaction_dt: number;
  transaction_amt: number;
  product_cd?: string;
  card1?: number;
  card4?: string;
  card6?: string;
  addr1?: number;
  addr2?: number;
  dist1?: number;
  p_emaildomain?: string;
  device_type?: string;
  device_info?: string;
  extra_features?: Record<string, number | string | null>;
}

export interface MlRiskIndicator {
  feature: string;
  value: number | string;
  importance: number | null;
  signal: string;
}

export interface MlPredictionResponse {
  transaction_id: string;
  xgboost_probability: number;
  gnn_probability: number | null;
  final_score: number;
  decision: MlDecision;
  inference_latency_ms: number;
  model_version: string;
  scoring_mode: 'xgboost_only' | 'hybrid_xgb_gnn';
  gnn_available: boolean;
  thresholds: {
    review: number;
    block: number;
  };
  risk_indicators: MlRiskIndicator[];
  scored_at_utc: string;
  source: 'xgboost_inference';
}

export interface MlHealthStatus {
  status: 'ok' | 'degraded';
  service: string;
  model_ready: boolean;
  model_version: string | null;
  missing_artifacts: string[];
  uptime_seconds: number;
  timestamp_utc: string;
}

export interface MlModelInfo {
  ready: boolean;
  model_type: string;
  model_version: string | null;
  trained_at_utc: string | null;
  artifact_dir: string;
  missing_artifacts: string[];
  feature_count: number | null;
  feature_names: string[];
  thresholds: {
    review: number;
    block: number;
  } | null;
  validation_metrics: Record<string, unknown> | null;
  pr_curve: Array<{ recall: number; precision: number; threshold?: number }> | null;
  gnn_status: 'not_implemented_phase4' | 'unavailable' | 'ready';
}

export interface MlApiError {
  status: number;
  error: string;
  detail: string;
  missing_artifacts?: string[];
}

