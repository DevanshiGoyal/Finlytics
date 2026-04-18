export type CreditDemandScenario = "baseline" | "optimistic" | "pessimistic";

export interface CreditDemandByGradeResponse {
  trend: Array<{
    month: string;
    A: number;
    B: number;
    C: number;
    D: number;
    E: number;
  }>;
  heatmap: Array<{ grade: string; values: number[] }>;
}

export interface CreditDemandForecastRequest {
  horizon: number;
  confidence: number;
  scenarioType: CreditDemandScenario;
  baseVolume: number | null;
}

export interface CreditDemandForecast {
  grade: string;
  predictions: Array<{
    month: string;
    forecast_central: number;
    forecast_lower: number;
    forecast_upper: number;
    historical: number | null;
  }>;
  modelMetrics: {
    mape: number;
    rmse: number;
    trainingMapeOnTestSet: number;
  };
  featureImportance: Array<{
    feature: string;
    importance: number;
  }>;
  modelName?: string;
}

export interface CreditDemandForecastResponse {
  forecasts: CreditDemandForecast[];
  metadata: {
    horizon: number;
    confidence: number;
    scenario: CreditDemandScenario;
    totalForecastedVolume: number;
    scenarioComparison?: {
      baseline: number;
      optimistic: number;
      pessimistic: number;
    };
  };
  warnings?: string[];
}
