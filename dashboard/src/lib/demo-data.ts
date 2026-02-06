/**
 * Self-contained demo data generator for the static dashboard.
 * Uses a seeded PRNG so charts are deterministic across page loads
 * but still look realistic.
 */

// Seeded pseudo-random (mulberry32)
function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seededGauss(rng: () => number, mean: number, std: number): number {
  const u1 = rng();
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1 || 0.001)) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}

const NOW = new Date();
const HOUR = 3600_000;
const DAY = 86400_000;

// ─── Grid Demand (Power) ────────────────────────────────────────────────────

export interface GridDemandPoint {
  timestamp: string;
  value: number;
  region: string;
}

export function generateGridDemand(
  region: string,
  days: number = 7
): GridDemandPoint[] {
  const rng = mulberry32(region === "PJM" ? 42 : 99);
  const baseLoad = region === "PJM" ? 85000 : 45000;
  const points: GridDemandPoint[] = [];

  for (let h = days * 24 - 1; h >= 0; h--) {
    const ts = new Date(NOW.getTime() - h * HOUR);
    const hour = ts.getUTCHours();
    const dailyFactor = 1.0 + 0.15 * Math.max(0, (hour - 6) * (18 - hour)) / 36;
    const trend = 1.0 + 0.001 * (days * 24 - h);
    const noise = seededGauss(rng, 0, 0.02);
    const val = baseLoad * dailyFactor * trend * (1 + noise);

    points.push({
      timestamp: ts.toISOString(),
      value: Math.round(val * 10) / 10,
      region,
    });
  }
  return points;
}

// ─── Night Base Load ────────────────────────────────────────────────────────

export interface BaseLoadPoint {
  date: string;
  base_load_mw: number;
}

export function generateBaseLoad(
  region: string,
  days: number = 30
): BaseLoadPoint[] {
  const demand = generateGridDemand(region, days);
  const byDate: Record<string, number[]> = {};

  for (const p of demand) {
    const d = new Date(p.timestamp);
    const hour = d.getUTCHours();
    if (hour >= 2 && hour <= 4) {
      const dateKey = d.toISOString().split("T")[0];
      if (!byDate[dateKey]) byDate[dateKey] = [];
      byDate[dateKey].push(p.value);
    }
  }

  return Object.entries(byDate)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, vals]) => {
      vals.sort((a, b) => a - b);
      const median = vals[Math.floor(vals.length / 2)];
      return { date, base_load_mw: Math.round(median * 10) / 10 };
    });
}

// ─── GPU Spot Pricing ───────────────────────────────────────────────────────

export interface GPUSpotPoint {
  timestamp: string;
  value: number;
  gpu: string;
  instance_type: string;
}

const GPU_CONFIGS = [
  { type: "p5.48xlarge", gpu: "H100", base: 12.5, vol: 0.12 },
  { type: "p4d.24xlarge", gpu: "A100", base: 4.5, vol: 0.10 },
  { type: "g5.48xlarge", gpu: "A10G", base: 1.8, vol: 0.08 },
  { type: "g6.48xlarge", gpu: "L4", base: 1.2, vol: 0.07 },
];

export function generateGPUSpotPricing(days: number = 7): GPUSpotPoint[] {
  const points: GPUSpotPoint[] = [];

  for (const cfg of GPU_CONFIGS) {
    const rng = mulberry32(cfg.base * 1000);
    for (let h = days * 24 - 1; h >= 0; h--) {
      const ts = new Date(NOW.getTime() - h * HOUR);
      const trend = 1.0 + 0.0005 * (days * 24 - h);
      const noise = seededGauss(rng, 0, cfg.vol);
      const val = Math.max(0.5, cfg.base * trend * (1 + noise));
      points.push({
        timestamp: ts.toISOString(),
        value: Math.round(val * 10000) / 10000,
        gpu: cfg.gpu,
        instance_type: cfg.type,
      });
    }
  }
  return points;
}

// ─── Token Pricing ──────────────────────────────────────────────────────────

export interface TokenPriceEntry {
  provider: string;
  model: string;
  input_price: number;
  output_price: number;
  deflation_pct: number;
}

export function getTokenPricing(): TokenPriceEntry[] {
  return [
    { provider: "OpenAI", model: "GPT-4o", input_price: 2.50, output_price: 10.00, deflation_pct: -15.2 },
    { provider: "OpenAI", model: "GPT-4o-mini", input_price: 0.15, output_price: 0.60, deflation_pct: -25.0 },
    { provider: "OpenAI", model: "o1", input_price: 15.00, output_price: 60.00, deflation_pct: 0.0 },
    { provider: "Anthropic", model: "Claude Sonnet 4", input_price: 3.00, output_price: 15.00, deflation_pct: -10.5 },
    { provider: "Anthropic", model: "Claude Haiku 3.5", input_price: 0.80, output_price: 4.00, deflation_pct: -20.0 },
    { provider: "Google", model: "Gemini 2.0 Flash", input_price: 0.10, output_price: 0.40, deflation_pct: -40.0 },
    { provider: "Google", model: "Gemini 1.5 Pro", input_price: 1.25, output_price: 5.00, deflation_pct: -18.3 },
    { provider: "Amazon", model: "Nova Pro", input_price: 0.80, output_price: 3.20, deflation_pct: -12.0 },
    { provider: "Amazon", model: "Nova Lite", input_price: 0.06, output_price: 0.24, deflation_pct: -33.3 },
  ];
}

// ─── Bandwidth / HTTP Traffic ───────────────────────────────────────────────

export interface TrafficPoint {
  timestamp: string;
  value: number;
}

export function generateTrafficData(days: number = 7): TrafficPoint[] {
  const rng = mulberry32(777);
  const points: TrafficPoint[] = [];
  const base = 1_200_000_000;

  for (let h = days * 24 - 1; h >= 0; h--) {
    const ts = new Date(NOW.getTime() - h * HOUR);
    const hour = ts.getUTCHours();
    const dailyFactor = 1.0 + 0.3 * Math.max(0, (hour - 8) * (22 - hour)) / 49;
    const trend = 1.0 + 0.002 * (days * 24 - h);
    const noise = seededGauss(rng, 0, 0.04);
    const val = base * dailyFactor * trend * (1 + noise);
    points.push({
      timestamp: ts.toISOString(),
      value: Math.round(val),
    });
  }
  return points;
}

// ─── Alerts ─────────────────────────────────────────────────────────────────

export interface AlertEvent {
  id: number;
  alert_type: string;
  fired_at: string;
  severity: string;
  message: string;
}

export function generateAlerts(): AlertEvent[] {
  return [
    {
      id: 1,
      alert_type: "tightness_signal",
      fired_at: new Date(NOW.getTime() - 2 * HOUR).toISOString(),
      severity: "info",
      message: "GPU spot prices rising (slope=+0.0312/hr over 14d). Compute tightness increasing.",
    },
    {
      id: 2,
      alert_type: "inference_ramp",
      fired_at: new Date(NOW.getTime() - 18 * HOUR).toISOString(),
      severity: "info",
      message: "Bandwidth growing faster than power demand — inference monetization signal.",
    },
    {
      id: 3,
      alert_type: "overbuild_signal",
      fired_at: new Date(NOW.getTime() - 3 * DAY).toISOString(),
      severity: "warning",
      message: "GPU A10G spot prices declining (slope=-0.0045/hr over 14d). Potential mid-tier overbuild.",
    },
  ];
}

// ─── Connector Status ───────────────────────────────────────────────────────

export interface ConnectorStatus {
  name: string;
  last_run: string;
  status: string;
  rows_ingested: number;
}

export function getConnectorStatus(): ConnectorStatus[] {
  return [
    { name: "eia_grid", last_run: new Date(NOW.getTime() - 45 * 60000).toISOString(), status: "success", rows_ingested: 336 },
    { name: "cloudflare_radar", last_run: new Date(NOW.getTime() - 12 * 60000).toISOString(), status: "success", rows_ingested: 168 },
    { name: "token_pricing", last_run: new Date(NOW.getTime() - 6 * HOUR).toISOString(), status: "success", rows_ingested: 18 },
    { name: "gpu_spot", last_run: new Date(NOW.getTime() - 28 * 60000).toISOString(), status: "success", rows_ingested: 672 },
  ];
}

// ─── Composite Score ────────────────────────────────────────────────────────

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

export function computeScore(): CompositeScore {
  const components: Record<string, ScoreComponent> = {
    power_baseload: { score: 0.62, weight: 25, data_points: 336, confidence: 0.9 },
    gpu_spot_tightness: { score: 0.71, weight: 15, data_points: 672, confidence: 0.85 },
    token_deflation: { score: 0.68, weight: 15, data_points: 18, confidence: 0.85 },
    bandwidth_proxy: { score: 0.58, weight: 10, data_points: 168, confidence: 0.7 },
    water_anomaly: { score: 0.50, weight: 10, data_points: 0, confidence: 0.0 },
    labor_ops_ramp: { score: 0.50, weight: 10, data_points: 0, confidence: 0.0 },
    shadow_price_spread: { score: 0.50, weight: 10, data_points: 0, confidence: 0.0 },
    pue_cooling: { score: 0.50, weight: 5, data_points: 0, confidence: 0.0 },
  };

  let totalWeighted = 0;
  let totalWeight = 0;
  for (const comp of Object.values(components)) {
    const effectiveWeight = comp.weight * comp.confidence;
    totalWeighted += comp.score * effectiveWeight;
    totalWeight += effectiveWeight;
  }

  const composite = totalWeight > 0
    ? Math.round(Math.min(100, Math.max(0, (totalWeighted / totalWeight) * 100)) * 10) / 10
    : 50;

  return {
    utilization_confidence: composite,
    components,
    computed_at: NOW.toISOString(),
  };
}

// ─── Ops/Labor (stub for v2 panel) ─────────────────────────────────────────

export interface LaborPoint {
  date: string;
  ops_roles: number;
  construction_roles: number;
  cooling_roles: number;
}

export function generateLaborData(days: number = 60): LaborPoint[] {
  const rng = mulberry32(555);
  const points: LaborPoint[] = [];

  for (let d = days - 1; d >= 0; d--) {
    const ts = new Date(NOW.getTime() - d * DAY);
    const trend = 1.0 + 0.005 * (days - d);
    points.push({
      date: ts.toISOString().split("T")[0],
      ops_roles: Math.round(120 * trend * (1 + seededGauss(rng, 0, 0.08))),
      construction_roles: Math.round(85 * (1 + seededGauss(rng, 0, 0.06))),
      cooling_roles: Math.round(35 * trend * 1.2 * (1 + seededGauss(rng, 0, 0.1))),
    });
  }
  return points;
}

// ─── Quarterly Financials (stub) ────────────────────────────────────────────

export interface QuarterlyFinancial {
  quarter: string;
  provider: string;
  ai_capex_b: number;
  ai_revenue_b: number;
  conversion_velocity: number;
}

export function getQuarterlyFinancials(): QuarterlyFinancial[] {
  return [
    { quarter: "Q1 2025", provider: "Amazon/AWS", ai_capex_b: 21.4, ai_revenue_b: 28.8, conversion_velocity: 1.35 },
    { quarter: "Q1 2025", provider: "Microsoft", ai_capex_b: 16.8, ai_revenue_b: 22.1, conversion_velocity: 1.32 },
    { quarter: "Q1 2025", provider: "Google", ai_capex_b: 17.2, ai_revenue_b: 12.3, conversion_velocity: 0.72 },
    { quarter: "Q4 2024", provider: "Amazon/AWS", ai_capex_b: 18.2, ai_revenue_b: 26.3, conversion_velocity: 1.44 },
    { quarter: "Q4 2024", provider: "Microsoft", ai_capex_b: 14.9, ai_revenue_b: 19.8, conversion_velocity: 1.33 },
    { quarter: "Q4 2024", provider: "Google", ai_capex_b: 14.3, ai_revenue_b: 10.1, conversion_velocity: 0.71 },
    { quarter: "Q3 2024", provider: "Amazon/AWS", ai_capex_b: 15.7, ai_revenue_b: 23.1, conversion_velocity: 1.47 },
    { quarter: "Q3 2024", provider: "Microsoft", ai_capex_b: 13.1, ai_revenue_b: 17.2, conversion_velocity: 1.31 },
    { quarter: "Q3 2024", provider: "Google", ai_capex_b: 12.6, ai_revenue_b: 8.5, conversion_velocity: 0.67 },
  ];
}
