'use client';

import { useEffect, useState } from 'react';
import ReactEChartsCore from 'echarts-for-react';
import { api } from '@/lib/api';

export default function BandwidthPanel() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    api.getMetricSeries('net.http_requests', { days: '7', region: 'US' })
      .then(setData)
      .catch(() => {});
  }, []);

  const option = data ? {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis' as const,
      formatter: (params: any) => {
        const p = params[0];
        const val = p.value[1];
        const formatted = val >= 1e9 ? `${(val / 1e9).toFixed(2)}B`
          : val >= 1e6 ? `${(val / 1e6).toFixed(2)}M`
          : val.toLocaleString();
        return `${p.axisValueLabel}<br/>${formatted} requests`;
      },
    },
    xAxis: {
      type: 'time' as const,
      axisLabel: { color: '#a1a1aa', fontSize: 10 },
      axisLine: { lineStyle: { color: '#2a2a3a' } },
    },
    yAxis: {
      type: 'value' as const,
      name: 'Requests',
      nameTextStyle: { color: '#a1a1aa' },
      axisLabel: {
        color: '#a1a1aa',
        fontSize: 10,
        formatter: (v: number) => v >= 1e9 ? `${(v / 1e9).toFixed(1)}B` : `${(v / 1e6).toFixed(0)}M`,
      },
      splitLine: { lineStyle: { color: '#1a1a25' } },
    },
    series: [{
      type: 'line',
      data: data.data.map((d: any) => [d.timestamp, d.value]),
      smooth: true,
      symbol: 'none',
      lineStyle: { width: 1.5, color: '#a855f7' },
      areaStyle: { color: 'rgba(168,85,247,0.1)' },
    }],
    grid: { left: 70, right: 20, top: 30, bottom: 30 },
  } : null;

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4">Bandwidth / Inference Exhaust</h2>
      <h3 className="text-sm text-zinc-400 mb-2">HTTP Traffic Trends - US (7d)</h3>
      {option ? (
        <ReactEChartsCore option={option} style={{ height: 280 }} />
      ) : (
        <div className="h-[280px] animate-pulse bg-[#1a1a25] rounded" />
      )}
    </div>
  );
}
