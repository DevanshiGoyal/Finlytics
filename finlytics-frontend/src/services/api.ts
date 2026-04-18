import type {
  CreditDemandByGradeResponse,
  CreditDemandForecastRequest,
  CreditDemandForecastResponse,
} from "./types";

export type {
  CreditDemandByGradeResponse,
  CreditDemandForecast,
  CreditDemandForecastRequest,
  CreditDemandForecastResponse,
  CreditDemandScenario,
} from "./types";

type JsonBody = Record<string, unknown>;

function normaliseBaseUrl(url: string): string {
  return url.trim().replace(/\/+$/, "");
}

const DEFAULT_LOCAL_TALK_TO_DATA_API_BASE = "http://127.0.0.1:8000";
const API_BASE = normaliseBaseUrl(process.env.NEXT_PUBLIC_API_BASE_URL || "");
const TALK_TO_DATA_API_BASE = normaliseBaseUrl(
  process.env.NEXT_PUBLIC_TALK_TO_DATA_API_BASE_URL ||
    process.env.NEXT_PUBLIC_API_BASE_URL ||
    (process.env.NODE_ENV === "development" ? DEFAULT_LOCAL_TALK_TO_DATA_API_BASE : "")
);

type ContentTypeMode = "json" | "none";

function parseErrorMessage(status: number, raw: string): string {
  let message = `API request failed: ${status}`;

  if (!raw) {
    return message;
  }

  const trimmed = raw.trim();
  if (/^<!doctype html>/i.test(trimmed) || /^<html/i.test(trimmed)) {
    return `${message} - received an HTML page instead of API JSON. Check NEXT_PUBLIC_API_BASE_URL (or NEXT_PUBLIC_TALK_TO_DATA_API_BASE_URL).`;
  }

  try {
    const parsed = JSON.parse(trimmed) as {
      detail?: unknown;
      error?: string;
      message?: string;
    };

    if (typeof parsed.detail === "string" && parsed.detail.trim()) {
      return parsed.detail;
    }

    if (typeof parsed.error === "string" && parsed.error.trim()) {
      return parsed.error;
    }

    if (typeof parsed.message === "string" && parsed.message.trim()) {
      return parsed.message;
    }

    if (parsed.detail) {
      return `${message} - ${JSON.stringify(parsed.detail)}`;
    }

    return message;
  } catch {
    return `${message} - ${raw}`;
  }
}

async function requestWithBase<T>(
  baseUrl: string,
  endpoint: string,
  options?: RequestInit,
  contentType: ContentTypeMode = "json"
): Promise<T> {
  const finalBase = normaliseBaseUrl(baseUrl);
  const finalEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;

  const headers = new Headers(options?.headers);
  if (contentType === "json" && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  let response: Response;
  try {
    response = await fetch(`${finalBase}${finalEndpoint}`, {
      ...options,
      headers,
    });
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(
        `Network error while calling ${finalBase}${finalEndpoint}. Ensure the backend is running and allows requests from this frontend origin (CORS).`
      );
    }

    throw error;
  }

  if (!response.ok) {
    const raw = await response.text();
    throw new Error(parseErrorMessage(response.status, raw));
  }

  return (await response.json()) as T;
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  return requestWithBase<T>(API_BASE, endpoint, options, "json");
}

async function requestTalkToData<T>(
  endpoint: string,
  options?: RequestInit,
  contentType: ContentTypeMode = "json"
): Promise<T> {
  return requestWithBase<T>(TALK_TO_DATA_API_BASE, endpoint, options, contentType);
}

export type FeatureImportance = Array<{ feature: string; importance: number }>;

export interface ShapWaterfallPoint {
  feature: string;
  value: number;
  shapValue: number;
  start: number;
  end: number;
}

export interface ShapExplanationResponse {
  available: boolean;
  message: string | null;
  baseValue: number | null;
  modelOutput: number | null;
  points: ShapWaterfallPoint[];
}

export interface DepositLeaderboardResponse {
  leaderboard: Array<{
    model: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
  }>;
  featureImportance: FeatureImportance;
  bestModel: string;
}

export interface DepositPredictionResponse {
  probability: number;
  label: string;
  prediction: number;
  model: string;
  shapExplanation?: ShapExplanationResponse;
}

export interface DefaultPredictionResponse {
  probability: number;
  label: string;
  explainability: FeatureImportance;
  shapExplanation?: ShapExplanationResponse;
}

export interface ChurnPredictionResponse {
  probability: number;
  label: string;
  suggestions: string[];
  explainability: FeatureImportance;
  shapExplanation?: ShapExplanationResponse;
}

export type LoanForecastScenario = "baseline" | "optimistic" | "conservative" | "stress";

export interface LoanForecastRequest {
  horizonMonths: number;
  scenario: LoanForecastScenario;
  growthAdjustmentPct: number;
  interestRateShockBps: number;
  loanCountChangePct: number;
  avgLoanAmountChangePct: number;
}

export interface LoanForecastSummary {
  horizonMonths: number;
  scenario: string;
  projectedTotal: number;
  averageMonthly: number;
  finalMonth: string;
  finalValue: number;
  finalRange: {
    low: number;
    high: number;
  };
  growthVsLastActualPct: number;
  assumptions: {
    growthAdjustmentPct: number;
    interestRateShockBps: number;
    loanCountChangePct: number;
    avgLoanAmountChangePct: number;
  };
}

export interface LoanForecastModelInfo {
  primary: string;
  blend: string;
  scaler: string;
}

export interface LoanForecastResponse {
  trend: Array<{ month: string; actual: number; forecast: number }>;
  interval: Array<{
    month: string;
    value: number;
    low: number;
    high: number;
    historical: number | null;
  }>;
  summary?: LoanForecastSummary;
  request?: LoanForecastRequest & { userAdjustments?: Partial<LoanForecastRequest> };
  model?: LoanForecastModelInfo;
  warnings?: string[];
}

export interface AnomalyDetectResponse {
  score: number;
  label: string;
  shapExplanation?: ShapExplanationResponse;
}

export interface AnomalyTimeseriesResponse {
  trend: Array<{ time: string; score: number }>;
  batch: Array<{ txId: string; amount: number; score: number; label: string }>;
}

export interface AnomalyBatchScoreResponse {
  batch: Array<{ txId: string; amount: number; score: number; label: string }>;
}

export interface WealthPersonaSummary {
  dataset_source: string;
  datasetPath: string;
  kaggle_dataset_url: string;
  rfm_notebook_url: string;
  persona_notebook_url: string;
  behavior_notebook_url: string;
  transaction_rows: number;
  customers: number;
  vital_few_customers: number;
  vital_few_share: number;
  behavioral_anomalies: number;
  high_potential_zones: number;
  k: number;
  topN: number;
}

export interface WealthPersonaBreakdownRow {
  [key: string]: unknown;
  cluster: number;
  persona: string;
  customers: number;
  medianAge: number;
  avgRecency: number;
  avgFrequency: number;
  avgMonetary: number;
  avgAccountBalance: number;
  vitalFewShare: number;
}

export interface WealthPersonaCountRow {
  [key: string]: unknown;
  persona: string;
  customers: number;
}

export interface WealthVitalFewRow {
  [key: string]: unknown;
  CustomerID: string;
  persona: string;
  location: string;
  rfmScore: string;
  wealthValue: number;
  frequency: number;
  monetary: number;
  avgAccountBalance: number;
  medianAge: number;
}

export interface WealthRegionRow {
  [key: string]: unknown;
  location: string;
  customers: number;
  totalAccountBalance: number;
  avgAccountBalance: number;
  totalTransactionValue: number;
  vitalFewShare: number;
  potentialScore: number;
  highPotentialZone: boolean;
}

export interface WealthBehavioralAnomalyRow {
  [key: string]: unknown;
  customerId: string;
  transactionDate: string;
  transactionAmount: number;
  location: string;
  anomalyScore: number;
  reason: string;
}

export interface WealthPersonaInsightsResponse {
  summary: WealthPersonaSummary;
  personaBreakdown: WealthPersonaBreakdownRow[];
  personaCounts: WealthPersonaCountRow[];
  topVitalFew: WealthVitalFewRow[];
  topRegions: WealthRegionRow[];
  topBehavioralAnomalies: WealthBehavioralAnomalyRow[];
}

export async function getDepositLeaderboard() {
  return request<DepositLeaderboardResponse>("/deposit/leaderboard", {
    method: "GET",
  });
}

export async function predictDeposit(payload: JsonBody) {
  return request<DepositPredictionResponse>("/deposit/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function predictDefault(payload: JsonBody) {
  return request<DefaultPredictionResponse>("/predict/default", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function predictChurn(payload: JsonBody) {
  return request<ChurnPredictionResponse>("/predict/churn", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getLoanVolumeForecast() {
  return request<LoanForecastResponse>("/forecast/loan-volume", {
    method: "GET",
  });
}

export async function predictLoanVolumeForecast(payload: LoanForecastRequest) {
  return request<LoanForecastResponse>("/forecast/loan-volume", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function detectAnomaly(payload: JsonBody) {
  return request<AnomalyDetectResponse>("/anomaly/detect", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getAnomalyTimeseries() {
  return request<AnomalyTimeseriesResponse>("/anomaly/timeseries", {
    method: "GET",
  });
}

export async function scoreAnomalyBatch(payload: JsonBody) {
  return request<AnomalyBatchScoreResponse>("/anomaly/batch-score", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getWealthPersonaInsights(payload?: {
  datasetPath?: string;
  k?: number;
  topN?: number;
}) {
  return request<WealthPersonaInsightsResponse>("/wealth-persona/insights", {
    method: "POST",
    body: JSON.stringify(payload ?? {}),
  });
}

export async function getCreditDemandByGrade() {
  return request<CreditDemandByGradeResponse>("/credit-demand/by-grade", {
    method: "GET",
  });
}

export async function forecastCreditDemandByGrade(payload: CreditDemandForecastRequest) {
  return request<CreditDemandForecastResponse>("/credit-demand/by-grade", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export type TalkToDataMode = "raw" | "smart" | "scalable";

export interface TalkToDataColumnInfo {
  name: string;
  type: string;
  null_pct: number;
  mean?: number | null;
  min?: number | null;
  max?: number | null;
  unique_count?: number | null;
}

export interface TalkToDataUploadResponse {
  dataset_id: string;
  filename: string;
  row_count: number;
  columns: TalkToDataColumnInfo[];
  sample: Array<Record<string, unknown>>;
  suggested_questions: string[];
}

export interface TalkToDataHealth {
  missing_pct: number;
  outliers: number;
  rows_used: number;
  confidence: number;
  confidence_level?: string;
  confidence_reason?: string[];
  summary_text?: string;
  penalty_breakdown?: Record<string, number>;
}

export interface TalkToDataQueryResponse {
  sql: string;
  result: Array<Record<string, unknown>>;
  columns: string[];
  explanation: string;
  insights: string[];
  chart_type?: string | null;
  chart_x?: string | null;
  chart_y: string[];
  data_health: TalkToDataHealth;
  preprocessing_log: string[];
  mode: TalkToDataMode;
  why_analysis?: string | null;
  error?: string | null;
}

export interface TalkToDataChartDataset {
  chart_type: string;
  chart_x: string;
  chart_y: string[];
  result: Array<Record<string, unknown>>;
  title: string;
  sql: string;
}

export interface TalkToDataAutoVisualizeResponse {
  trend?: TalkToDataChartDataset | null;
  composition?: TalkToDataChartDataset | null;
  comparison?: TalkToDataChartDataset | null;
  summary_stats?: Record<string, unknown>;
}

export interface TalkToDataCorrelationPoint {
  col_a: string;
  col_b: string;
  correlation: number | null;
}

export interface TalkToDataCorrelationResponse {
  columns: string[];
  data: TalkToDataCorrelationPoint[];
  method: string;
  note: string;
}

export async function uploadTalkToDataCSV(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  return requestTalkToData<TalkToDataUploadResponse>("/upload", {
    method: "POST",
    body: formData,
  }, "none");
}

export async function queryTalkToData(payload: {
  datasetId: string;
  question: string;
  mode: TalkToDataMode;
  sessionId?: string;
}) {
  return requestTalkToData<TalkToDataQueryResponse>("/query", {
    method: "POST",
    body: JSON.stringify({
      dataset_id: payload.datasetId,
      question: payload.question,
      mode: payload.mode,
      session_id: payload.sessionId,
    }),
  });
}

export async function getTalkToDataHealth(payload: { datasetId: string; mode: TalkToDataMode }) {
  return requestTalkToData<TalkToDataHealth>("/data-health", {
    method: "POST",
    body: JSON.stringify({
      dataset_id: payload.datasetId,
      mode: payload.mode,
    }),
  });
}

export async function getTalkToDataAutoVisualize(payload: {
  datasetId: string;
  mode: TalkToDataMode;
}) {
  return requestTalkToData<TalkToDataAutoVisualizeResponse>("/auto-visualize", {
    method: "POST",
    body: JSON.stringify({
      dataset_id: payload.datasetId,
      mode: payload.mode,
    }),
  });
}

export async function getTalkToDataCorrelationMatrix(payload: {
  datasetId: string;
  method?: "pearson" | "spearman" | "kendall";
}) {
  return requestTalkToData<TalkToDataCorrelationResponse>("/correlation-matrix", {
    method: "POST",
    body: JSON.stringify({
      dataset_id: payload.datasetId,
      method: payload.method ?? "pearson",
    }),
  });
}
