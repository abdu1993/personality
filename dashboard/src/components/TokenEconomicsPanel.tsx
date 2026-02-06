'use client';

import ReactEChartsCore from 'echarts-for-react';
import { getTokenPricing } from '@/lib/demo-data';

const PROVIDER_COLORS: Record<string, string> = {
  OpenAI: '#10b981',
  Anthropic: '#d97706',
  Google: '#3b82f6',
  Amazon: '#ef4444',
};

export default function TokenEconomicsPanel() {
  const data = getTokenPricing();
  const sorted = [...data].sort((a, b) => b.input_price - a.input_price);

  const chartOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis' as const,
      backgroundColor: '#1a1a25',
      borderColor: '#2a2a3a',
      textStyle: { color: '#e4e4e7', fontSize: 12 },
    },
    xAxis: {
      type: 'category' as const,
      data: sorted.map(d => d.model),
      axisLabel: { color: '#71717a', fontSize: 9, rotate: 35 },
      axisLine: { lineStyle: { color: '#2a2a3a' } },
    },
    yAxis: {
      type: 'value' as const,
      name: '$/1M tokens',
      nameTextStyle: { color: '#71717a', fontSize: 10 },
      axisLabel: { color: '#71717a', fontSize: 10 },
      splitLine: { lineStyle: { color: '#1a1a25' } },
    },
    series: [
      {
        name: 'Input',
        type: 'bar',
        data: sorted.map(d => ({
          value: d.input_price,
          itemStyle: { color: PROVIDER_COLORS[d.provider] || '#888', opacity: 0.9 },
        })),
        barGap: '10%',
      },
      {
        name: 'Output',
        type: 'bar',
        data: sorted.map(d => ({
          value: d.output_price,
          itemStyle: { color: PROVIDER_COLORS[d.provider] || '#888', opacity: 0.5 },
        })),
      },
    ],
    legend: {
      data: ['Input', 'Output'],
      textStyle: { color: '#a1a1aa', fontSize: 11 },
      top: 0,
    },
    grid: { left: 55, right: 15, top: 30, bottom: 65 },
  };

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-1">Token Economics</h2>
      <p className="text-xs text-zinc-500 mb-4">API pricing and deflation trends</p>
      <ReactEChartsCore option={chartOption} style={{ height: 280 }} />
      <div className="mt-3 space-y-1.5 max-h-[200px] overflow-y-auto">
        {data.map((d, i) => (
          <div key={i} className="inner-card flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: PROVIDER_COLORS[d.provider] || '#888' }} />
              <span className="text-sm">{d.model}</span>
              <span className="text-xs text-zinc-600">{d.provider}</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-zinc-400">${d.input_price} in / ${d.output_price} out</span>
              <span className={`text-xs font-semibold min-w-[50px] text-right ${d.deflation_pct <= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {d.deflation_pct > 0 ? '+' : ''}{d.deflation_pct}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
