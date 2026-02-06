'use client';

import { useEffect, useState } from 'react';
import ReactEChartsCore from 'echarts-for-react';
import { api } from '@/lib/api';

const PROVIDER_COLORS: Record<string, string> = {
  OpenAI: '#10b981',
  Anthropic: '#d97706',
  Google: '#3b82f6',
  Amazon: '#ef4444',
};

export default function TokenEconomicsPanel() {
  const [inputData, setInputData] = useState<any>(null);
  const [deflationData, setDeflationData] = useState<any>(null);

  useEffect(() => {
    api.getMetricSeries('model_price.input_usd_per_1m_tokens', { days: '30' })
      .then(setInputData)
      .catch(() => {});
    api.getTokenDeflation(90)
      .then(setDeflationData)
      .catch(() => {});
  }, []);

  // Build bar chart for current prices by model
  const priceOption = inputData ? (() => {
    // Get latest price per model
    const latest: Record<string, { provider: string; value: number }> = {};
    for (const d of inputData.data) {
      const model = d.metadata?.model || 'Unknown';
      latest[model] = { provider: d.provider, value: d.value };
    }
    const entries = Object.entries(latest).sort((a, b) => b[1].value - a[1].value);

    return {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' as const },
      xAxis: {
        type: 'category' as const,
        data: entries.map(([m]) => m),
        axisLabel: { color: '#a1a1aa', fontSize: 10, rotate: 30 },
        axisLine: { lineStyle: { color: '#2a2a3a' } },
      },
      yAxis: {
        type: 'value' as const,
        name: '$/1M tokens',
        nameTextStyle: { color: '#a1a1aa' },
        axisLabel: { color: '#a1a1aa', fontSize: 10 },
        splitLine: { lineStyle: { color: '#1a1a25' } },
      },
      series: [{
        type: 'bar',
        data: entries.map(([_, d]) => ({
          value: d.value,
          itemStyle: { color: PROVIDER_COLORS[d.provider] || '#888' },
        })),
      }],
      grid: { left: 70, right: 20, top: 30, bottom: 70 },
    };
  })() : null;

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4">Token Economics</h2>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div>
          <h3 className="text-sm text-zinc-400 mb-2">Input Token Prices by Model</h3>
          {priceOption ? (
            <ReactEChartsCore option={priceOption} style={{ height: 280 }} />
          ) : (
            <div className="h-[280px] animate-pulse bg-[#1a1a25] rounded" />
          )}
        </div>
        <div>
          <h3 className="text-sm text-zinc-400 mb-2">Token Deflation (90d)</h3>
          {deflationData ? (
            <div className="space-y-2 max-h-[280px] overflow-y-auto">
              {deflationData.data.map((d: any, i: number) => (
                <div key={i} className="flex items-center justify-between bg-[#1a1a25] rounded px-3 py-2">
                  <div>
                    <span className="text-sm font-medium">{d.model}</span>
                    <span className="text-xs text-zinc-500 ml-2">{d.provider}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-zinc-400">${d.latest_price}</span>
                    <span className={`text-sm font-semibold ${d.deflation_pct <= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {d.deflation_pct > 0 ? '+' : ''}{d.deflation_pct}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="h-[280px] animate-pulse bg-[#1a1a25] rounded" />
          )}
        </div>
      </div>
    </div>
  );
}
