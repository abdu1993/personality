'use client';

import { useMemo } from 'react';
import ReactEChartsCore from 'echarts-for-react';
import { generateGPUSpotPricing } from '@/lib/demo-data';

const GPU_COLORS: Record<string, string> = {
  H100: '#a855f7',
  A100: '#3b82f6',
  A10G: '#22c55e',
  L4: '#eab308',
};

export default function ComputeMarketPanel() {
  const data = useMemo(() => generateGPUSpotPricing(7), []);

  // Group by GPU type
  const grouped: Record<string, { ts: string[]; vals: number[] }> = {};
  for (const d of data) {
    if (!grouped[d.gpu]) grouped[d.gpu] = { ts: [], vals: [] };
    grouped[d.gpu].ts.push(d.timestamp);
    grouped[d.gpu].vals.push(d.value);
  }

  const series = Object.entries(grouped).map(([gpu, { ts, vals }]) => ({
    name: gpu,
    type: 'line' as const,
    data: ts.map((t, i) => [t, vals[i]]),
    smooth: true,
    symbol: 'none',
    lineStyle: { width: 1.5, color: GPU_COLORS[gpu] || '#888' },
    emphasis: { lineStyle: { width: 2.5 } },
  }));

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis' as const,
      backgroundColor: '#1a1a25',
      borderColor: '#2a2a3a',
      textStyle: { color: '#e4e4e7', fontSize: 12 },
    },
    legend: {
      data: Object.keys(grouped),
      textStyle: { color: '#a1a1aa', fontSize: 11 },
      top: 0,
      icon: 'roundRect',
      itemWidth: 14,
      itemHeight: 3,
    },
    xAxis: {
      type: 'time' as const,
      axisLabel: { color: '#71717a', fontSize: 10 },
      axisLine: { lineStyle: { color: '#2a2a3a' } },
    },
    yAxis: {
      type: 'value' as const,
      name: '$/hr',
      nameTextStyle: { color: '#71717a', fontSize: 10 },
      axisLabel: { color: '#71717a', fontSize: 10 },
      splitLine: { lineStyle: { color: '#1a1a25' } },
    },
    series,
    grid: { left: 50, right: 15, top: 35, bottom: 25 },
  };

  // Compute summary stats
  const summaries = Object.entries(grouped).map(([gpu, { vals }]) => {
    const latest = vals[vals.length - 1];
    const earliest = vals[0];
    const change = ((latest - earliest) / earliest * 100);
    return { gpu, latest, change };
  });

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-1">Compute Market Tightness</h2>
      <p className="text-xs text-zinc-500 mb-4">GPU spot pricing as utilization proxy</p>
      <ReactEChartsCore option={option} style={{ height: 280 }} />
      <div className="grid grid-cols-4 gap-2 mt-3">
        {summaries.map(s => (
          <div key={s.gpu} className="inner-card text-center">
            <div className="text-xs text-zinc-500">{s.gpu}</div>
            <div className="text-sm font-semibold" style={{ color: GPU_COLORS[s.gpu] }}>
              ${s.latest.toFixed(2)}/hr
            </div>
            <div className={`text-xs font-medium ${s.change >= 0 ? 'text-red-400' : 'text-green-400'}`}>
              {s.change >= 0 ? '+' : ''}{s.change.toFixed(1)}% 7d
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
