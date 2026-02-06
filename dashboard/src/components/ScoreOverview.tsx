'use client';

import ReactEChartsCore from 'echarts-for-react';
import { computeScore } from '@/lib/demo-data';

const LABELS: Record<string, string> = {
  power_baseload: 'Power / Base Load',
  gpu_spot_tightness: 'GPU Spot Tightness',
  token_deflation: 'Token Deflation',
  bandwidth_proxy: 'Bandwidth Proxy',
  water_anomaly: 'Water Anomaly',
  labor_ops_ramp: 'Labor Ops Ramp',
  shadow_price_spread: 'Shadow Price',
  pue_cooling: 'PUE / Cooling',
};

const COLORS: Record<string, string> = {
  power_baseload: '#3b82f6',
  gpu_spot_tightness: '#a855f7',
  token_deflation: '#22c55e',
  bandwidth_proxy: '#f59e0b',
  water_anomaly: '#06b6d4',
  labor_ops_ramp: '#ec4899',
  shadow_price_spread: '#ef4444',
  pue_cooling: '#6366f1',
};

export default function ScoreOverview() {
  const score = computeScore();
  const cs = score.utilization_confidence;

  const gaugeOption = {
    series: [{
      type: 'gauge',
      startAngle: 200,
      endAngle: -20,
      min: 0,
      max: 100,
      splitNumber: 10,
      radius: '100%',
      itemStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 1, y2: 0,
          colorStops: [
            { offset: 0, color: '#ef4444' },
            { offset: 0.5, color: '#eab308' },
            { offset: 1, color: '#22c55e' },
          ],
        },
      },
      progress: { show: true, width: 18 },
      pointer: { show: false },
      axisLine: { lineStyle: { width: 18, color: [[1, '#1a1a25']] } },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: { show: false },
      title: { show: false },
      detail: {
        valueAnimation: true,
        fontSize: 42,
        fontWeight: 'bold',
        color: '#e4e4e7',
        offsetCenter: [0, '5%'],
        formatter: '{value}',
      },
      data: [{ value: cs }],
    }],
  };

  const components = Object.entries(score.components);

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-1">Utilization Confidence Score</h2>
      <p className="text-xs text-zinc-500 mb-4">Composite weighted signal across power, compute, pricing, and traffic</p>
      <div className="flex flex-col lg:flex-row gap-6 items-center lg:items-start">
        <div className="flex-shrink-0 w-56 h-44">
          <ReactEChartsCore option={gaugeOption} style={{ height: '100%', width: '100%' }} />
        </div>
        <div className="flex-1 w-full">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {components.map(([key, comp]) => {
              const pct = Math.round(comp.score * 100);
              const color = COLORS[key] || '#888';
              return (
                <div key={key} className="inner-card">
                  <div className="text-[11px] text-zinc-400 mb-1.5 truncate">{LABELS[key] || key}</div>
                  <div className="flex items-baseline gap-1.5">
                    <span className="text-xl font-bold" style={{ color: comp.confidence > 0 ? color : '#555' }}>
                      {pct}
                    </span>
                    <span className="text-[10px] text-zinc-600">/ 100</span>
                  </div>
                  <div className="w-full bg-[#2a2a3a] rounded-full h-1 mt-2">
                    <div
                      className="h-1 rounded-full transition-all"
                      style={{ width: `${pct}%`, backgroundColor: comp.confidence > 0 ? color : '#444' }}
                    />
                  </div>
                  <div className="flex justify-between mt-1">
                    <span className="text-[10px] text-zinc-600">wt: {comp.weight}</span>
                    {comp.confidence === 0 && <span className="text-[10px] text-zinc-600">v2</span>}
                    {comp.confidence > 0 && <span className="text-[10px] text-zinc-600">{comp.data_points} pts</span>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
