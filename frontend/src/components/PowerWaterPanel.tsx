'use client';

import { useEffect, useState } from 'react';
import ReactEChartsCore from 'echarts-for-react';
import { api } from '@/lib/api';

export default function PowerWaterPanel() {
  const [region, setRegion] = useState('PJM');
  const [demandData, setDemandData] = useState<any>(null);
  const [baseloadData, setBaseloadData] = useState<any>(null);

  useEffect(() => {
    api.getMetricSeries('grid.demand_mw', { region, days: '7' })
      .then(setDemandData)
      .catch(() => {});
    api.getBaseload(region, 30)
      .then(setBaseloadData)
      .catch(() => {});
  }, [region]);

  const demandOption = demandData ? {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' as const },
    xAxis: {
      type: 'time' as const,
      axisLabel: { color: '#a1a1aa', fontSize: 10 },
      axisLine: { lineStyle: { color: '#2a2a3a' } },
    },
    yAxis: {
      type: 'value' as const,
      name: 'MW',
      nameTextStyle: { color: '#a1a1aa' },
      axisLabel: { color: '#a1a1aa', fontSize: 10 },
      splitLine: { lineStyle: { color: '#1a1a25' } },
    },
    series: [{
      type: 'line',
      data: demandData.data.map((d: any) => [d.timestamp, d.value]),
      smooth: true,
      lineStyle: { width: 1.5, color: '#3b82f6' },
      areaStyle: { color: 'rgba(59,130,246,0.1)' },
      symbol: 'none',
    }],
    grid: { left: 60, right: 20, top: 30, bottom: 30 },
  } : null;

  const baseloadOption = baseloadData ? {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' as const },
    xAxis: {
      type: 'category' as const,
      data: baseloadData.data.map((d: any) => d.date),
      axisLabel: { color: '#a1a1aa', fontSize: 10, rotate: 45 },
      axisLine: { lineStyle: { color: '#2a2a3a' } },
    },
    yAxis: {
      type: 'value' as const,
      name: 'MW',
      nameTextStyle: { color: '#a1a1aa' },
      axisLabel: { color: '#a1a1aa', fontSize: 10 },
      splitLine: { lineStyle: { color: '#1a1a25' } },
    },
    series: [{
      type: 'bar',
      data: baseloadData.data.map((d: any) => d.base_load_mw),
      itemStyle: { color: '#22c55e' },
    }],
    grid: { left: 60, right: 20, top: 30, bottom: 60 },
  } : null;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Power & Water</h2>
        <select
          value={region}
          onChange={e => setRegion(e.target.value)}
          className="bg-[#1a1a25] border border-[#2a2a3a] rounded px-2 py-1 text-sm text-zinc-300"
        >
          <option value="PJM">PJM (Data Center Alley)</option>
          <option value="ERCOT">ERCOT (Texas)</option>
        </select>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div>
          <h3 className="text-sm text-zinc-400 mb-2">Grid Demand (7d)</h3>
          {demandOption ? (
            <ReactEChartsCore option={demandOption} style={{ height: 250 }} />
          ) : (
            <div className="h-[250px] animate-pulse bg-[#1a1a25] rounded" />
          )}
        </div>
        <div>
          <h3 className="text-sm text-zinc-400 mb-2">Night Base Load Trend (30d)</h3>
          {baseloadOption ? (
            <ReactEChartsCore option={baseloadOption} style={{ height: 250 }} />
          ) : (
            <div className="h-[250px] animate-pulse bg-[#1a1a25] rounded" />
          )}
        </div>
      </div>
    </div>
  );
}
