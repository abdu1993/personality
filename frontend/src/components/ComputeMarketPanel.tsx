'use client';

import { useEffect, useState } from 'react';
import ReactEChartsCore from 'echarts-for-react';
import { api } from '@/lib/api';

const GPU_COLORS: Record<string, string> = {
  H100: '#a855f7',
  A100: '#3b82f6',
  A10G: '#22c55e',
  L4: '#eab308',
};

export default function ComputeMarketPanel() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    api.getMetricSeries('gpu_spot.price_usd_per_hour', { days: '7' })
      .then(setData)
      .catch(() => {});
  }, []);

  if (!data) return <div className="card animate-pulse h-80" />;

  // Group by GPU type
  const grouped: Record<string, { timestamps: string[]; values: number[] }> = {};
  for (const d of data.data) {
    const gpu = d.metadata?.gpu || 'Unknown';
    if (!grouped[gpu]) grouped[gpu] = { timestamps: [], values: [] };
    grouped[gpu].timestamps.push(d.timestamp);
    grouped[gpu].values.push(d.value);
  }

  const series = Object.entries(grouped).map(([gpu, { timestamps, values }]) => ({
    name: gpu,
    type: 'line' as const,
    data: timestamps.map((t, i) => [t, values[i]]),
    smooth: true,
    symbol: 'none',
    lineStyle: { width: 1.5, color: GPU_COLORS[gpu] || '#888' },
  }));

  const option = {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' as const },
    legend: {
      data: Object.keys(grouped),
      textStyle: { color: '#a1a1aa' },
      top: 0,
    },
    xAxis: {
      type: 'time' as const,
      axisLabel: { color: '#a1a1aa', fontSize: 10 },
      axisLine: { lineStyle: { color: '#2a2a3a' } },
    },
    yAxis: {
      type: 'value' as const,
      name: '$/hr',
      nameTextStyle: { color: '#a1a1aa' },
      axisLabel: { color: '#a1a1aa', fontSize: 10 },
      splitLine: { lineStyle: { color: '#1a1a25' } },
    },
    series,
    grid: { left: 60, right: 20, top: 40, bottom: 30 },
  };

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4">Compute Market Tightness</h2>
      <h3 className="text-sm text-zinc-400 mb-2">GPU Spot Price History (7d)</h3>
      <ReactEChartsCore option={option} style={{ height: 300 }} />
    </div>
  );
}
