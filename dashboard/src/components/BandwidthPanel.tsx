'use client';

import { useMemo } from 'react';
import ReactEChartsCore from 'echarts-for-react';
import { generateTrafficData } from '@/lib/demo-data';

export default function BandwidthPanel() {
  const data = useMemo(() => generateTrafficData(7), []);

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis' as const,
      backgroundColor: '#1a1a25',
      borderColor: '#2a2a3a',
      textStyle: { color: '#e4e4e7', fontSize: 12 },
      formatter: (params: any) => {
        const p = params[0];
        const val = p.value[1];
        const fmt = val >= 1e9 ? `${(val / 1e9).toFixed(2)}B`
          : val >= 1e6 ? `${(val / 1e6).toFixed(1)}M`
          : val.toLocaleString();
        return `${new Date(p.value[0]).toLocaleString()}<br/>HTTP Requests: <b>${fmt}</b>`;
      },
    },
    xAxis: {
      type: 'time' as const,
      axisLabel: { color: '#71717a', fontSize: 10 },
      axisLine: { lineStyle: { color: '#2a2a3a' } },
    },
    yAxis: {
      type: 'value' as const,
      name: 'Requests',
      nameTextStyle: { color: '#71717a', fontSize: 10 },
      axisLabel: {
        color: '#71717a',
        fontSize: 10,
        formatter: (v: number) => v >= 1e9 ? `${(v / 1e9).toFixed(1)}B` : `${(v / 1e6).toFixed(0)}M`,
      },
      splitLine: { lineStyle: { color: '#1a1a25' } },
    },
    series: [{
      type: 'line',
      data: data.map(d => [d.timestamp, d.value]),
      smooth: true,
      symbol: 'none',
      lineStyle: { width: 1.5, color: '#a855f7' },
      areaStyle: {
        color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [
          { offset: 0, color: 'rgba(168,85,247,0.2)' },
          { offset: 1, color: 'rgba(168,85,247,0.02)' },
        ]},
      },
    }],
    grid: { left: 65, right: 15, top: 30, bottom: 25 },
  };

  // Compute trend
  const first = data[0]?.value || 0;
  const last = data[data.length - 1]?.value || 0;
  const trendPct = first > 0 ? ((last - first) / first * 100) : 0;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold">Bandwidth / Inference Exhaust</h2>
          <p className="text-xs text-zinc-500">HTTP traffic trends as egress proxy (US region)</p>
        </div>
        <div className="inner-card flex items-center gap-2">
          <span className="text-xs text-zinc-400">7d trend</span>
          <span className={`text-sm font-semibold ${trendPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {trendPct >= 0 ? '+' : ''}{trendPct.toFixed(1)}%
          </span>
        </div>
      </div>
      <ReactEChartsCore option={option} style={{ height: 280 }} />
    </div>
  );
}
