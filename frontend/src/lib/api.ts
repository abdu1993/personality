const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  // Time series
  getMetricSeries: (metricId: string, params?: Record<string, string>) => {
    const qs = new URLSearchParams({ metric_id: metricId, ...params }).toString();
    return fetchApi<any>(`/api/metrics/series?${qs}`);
  },

  getAvailableMetrics: () =>
    fetchApi<any>('/api/metrics/available'),

  // Derived
  getBaseload: (region = 'PJM', days = 30) =>
    fetchApi<any>(`/api/derived/baseload?region=${region}&days=${days}`),

  getTokenDeflation: (days = 90) =>
    fetchApi<any>(`/api/derived/token-deflation?days=${days}`),

  // Score
  getScore: () =>
    fetchApi<any>('/api/score'),

  getWeights: () =>
    fetchApi<any>('/api/score/weights'),

  updateWeights: (weights: Record<string, number>) =>
    fetchApi<any>('/api/score/weights', {
      method: 'PUT',
      body: JSON.stringify(weights),
    }),

  // Alerts
  getAlertRules: () =>
    fetchApi<any>('/api/alerts/rules'),

  getAlertEvents: (days = 30) =>
    fetchApi<any>(`/api/alerts/events?days=${days}`),

  // Connectors
  getConnectorStatus: () =>
    fetchApi<any>('/api/connectors/status'),

  runConnector: (name: string) =>
    fetchApi<any>(`/api/connectors/run/${name}`, { method: 'POST' }),
};

// SWR fetcher
export const fetcher = (url: string) => fetch(`${API_BASE}${url}`).then(r => r.json());
