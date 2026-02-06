'use client';

import { useEffect, useState } from 'react';
import ReactEChartsCore from 'echarts-for-react';
import { api } from '@/lib/api';
import { CompositeScore } from '@/types';

const COMPONENT_LABELS: Record<string, string> = {
  power_baseload: 'Power / Base Load',
  gpu_spot_tightness: 'GPU Spot Tightness',
  token_deflation: 'Token Deflation',
  bandwidth_proxy: 'Bandwidth Proxy',
  water_anomaly: 'Water Anomaly',
  labor_ops_ramp: 'Labor Ops Ramp',
  shadow_price_spread: 'Shadow Price',
  pue_cooling: 'PUE / Cooling',
};

export default function ScoreOverview() {
  const [score, setScore] = useState<CompositeScore | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.getScore().then(setScore).catch(e => setError(e.message));
    const interval = setInterval(() => {
      api.getScore().then(setScore).catch(() => {});
    }, 300000); // 5 min
    return () => clearInterval(interval);
  }, []);

  if (error) return <div className="card text-red-400">Error loading score: {error}</div>;
  if (!score) return <div className="card animate-pulse h-48"></div>;

  const cs = score.utilization_confidence;
  const scoreClass = cs >= 65 ? 'score-high' : cs >= 40 ? 'score-mid' : 'score-low';

  const gaugeOption = {
    series: [{
      type: 'gauge',
      startAngle: 200,
      endAngle: -20,
      min: 0,
      max: 100,
      splitNumber: 10,
      itemStyle: { color: cs >= 65 ? '#22c55e' : cs >= 40 ? '#eab308' : '#ef4444' },
      progress: { show: true, width: 20 },
      pointer: { show: false },
      axisLine: { lineStyle: { width: 20, color: [[1, '#2a2a3a']] } },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: { show: false },
      title: { show: false },
      detail: {
        valueAnimation: true,
        fontSize: 36,
        fontWeight: 'bold',
        color: '#e4e4e7',
        offsetCenter: [0, 0],
        formatter: '{value}',
      },
      data: [{ value: cs }],
    }],
  };

  const components = Object.entries(score.components);

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4">Utilization Confidence Score</h2>
      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-shrink-0 w-64 h-48 mx-auto lg:mx-0">
          <ReactEChartsCore option={gaugeOption} style={{ height: '100%', width: '100%' }} />
        </div>
        <div className="flex-1">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {components.map(([key, comp]) => (
              <div key={key} className="rounded-md bg-[#1a1a25] p-3">
                <div className="text-xs text-zinc-400 mb-1">{COMPONENT_LABELS[key] || key}</div>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-bold">{(comp.score * 100).toFixed(0)}</span>
                  <span className="text-xs text-zinc-500">w:{comp.weight}</span>
                </div>
                <div className="w-full bg-[#2a2a3a] rounded-full h-1.5 mt-1">
                  <div
                    className="h-1.5 rounded-full"
                    style={{
                      width: `${comp.score * 100}%`,
                      backgroundColor: comp.confidence > 0 ? '#3b82f6' : '#555',
                    }}
                  />
                </div>
                {comp.confidence === 0 && (
                  <span className="text-[10px] text-zinc-600">No data</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
