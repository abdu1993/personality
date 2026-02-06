'use client';

import ReactEChartsCore from 'echarts-for-react';
import { getQuarterlyFinancials } from '@/lib/demo-data';

const PROVIDER_COLORS: Record<string, string> = {
  'Amazon/AWS': '#ef4444',
  Microsoft: '#3b82f6',
  Google: '#22c55e',
};

export default function QuarterlyPanel() {
  const data = getQuarterlyFinancials();
  const quarters = Array.from(new Set(data.map(d => d.quarter))).sort();
  const providers = Array.from(new Set(data.map(d => d.provider)));

  const capexSeries = providers.map(p => ({
    name: `${p} Capex`,
    type: 'bar' as const,
    data: quarters.map(q => {
      const row = data.find(d => d.quarter === q && d.provider === p);
      return row?.ai_capex_b || 0;
    }),
    itemStyle: { color: PROVIDER_COLORS[p] || '#888', opacity: 0.8 },
    stack: 'capex',
  }));

  const velocitySeries = providers.map(p => ({
    name: `${p} Velocity`,
    type: 'line' as const,
    yAxisIndex: 1,
    data: quarters.map(q => {
      const row = data.find(d => d.quarter === q && d.provider === p);
      return row?.conversion_velocity || 0;
    }),
    lineStyle: { width: 2, color: PROVIDER_COLORS[p] || '#888' },
    symbol: 'circle',
    symbolSize: 6,
    itemStyle: { color: PROVIDER_COLORS[p] || '#888' },
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
      data: providers.map(p => `${p} Capex`),
      textStyle: { color: '#a1a1aa', fontSize: 10 },
      top: 0,
      icon: 'roundRect',
      itemWidth: 12,
      itemHeight: 3,
    },
    xAxis: {
      type: 'category' as const,
      data: quarters,
      axisLabel: { color: '#71717a', fontSize: 10 },
      axisLine: { lineStyle: { color: '#2a2a3a' } },
    },
    yAxis: [
      {
        type: 'value' as const,
        name: 'Capex ($B)',
        nameTextStyle: { color: '#71717a', fontSize: 10 },
        axisLabel: { color: '#71717a', fontSize: 10 },
        splitLine: { lineStyle: { color: '#1a1a25' } },
      },
      {
        type: 'value' as const,
        name: 'Velocity',
        nameTextStyle: { color: '#71717a', fontSize: 10 },
        axisLabel: { color: '#71717a', fontSize: 10 },
        splitLine: { show: false },
      },
    ],
    series: [...capexSeries, ...velocitySeries],
    grid: { left: 50, right: 50, top: 35, bottom: 25 },
  };

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-1">Quarterly Fundamentals</h2>
      <p className="text-xs text-zinc-500 mb-4">AI Capex + Revenue Conversion Velocity (lines)</p>
      <ReactEChartsCore option={option} style={{ height: 280 }} />
    </div>
  );
}
