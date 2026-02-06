export interface MetricDataPoint {
  timestamp: string;
  value: number;
  unit: string;
  region: string;
  source: string;
  provider: string;
  confidence: number;
  metadata?: Record<string, any>;
}

export interface ScoreComponent {
  score: number;
  weight: number;
  data_points: number;
  confidence: number;
}

export interface CompositeScore {
  utilization_confidence: number;
  components: Record<string, ScoreComponent>;
  computed_at: string;
}

export interface AlertEvent {
  id: number;
  alert_type: string;
  fired_at: string;
  severity: string;
  message: string;
  details?: Record<string, any>;
  acknowledged: boolean;
}

export interface ConnectorStatus {
  name: string;
  last_run: string | null;
  status: string;
  rows_ingested: number;
  error: string | null;
}
