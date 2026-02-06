'use client';

import { useState, useMemo } from 'react';
import ReactEChartsCore from 'echarts-for-react';
import { generateGridDemand, generateBaseLoad } from '@/lib/demo-data';

export default function PowerWaterPanel() {
  const [region, setRegion] = useState('PJM');

  const demandData = useMemo(() => generateGridDemand(region, 7), [region]);
  const baseloadData = useMemo(() => generateBaseLoad(region, 30), [region]);

  const demandOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis' as const,
      backgroundColor: '#1a1a25',
      borderColor: '#2a2a3a',
      textStyle: { color: '#e4e4e7', fontSize: 12 },
    },
    xAxis: {
      type: 'time' as const,
      axisLabel: { color: '#71717a', fontSize: 10 },
      axisLine: { lineStyle: { color: '#2a2a3a' } },
    },
    yAxis: {
      type: 'value' as const,
      name: 'MW',
      nameTextStyle: { color: '#71717a', fontSize: 10 },
      axisLabel: { color: '#71717a', fontSize: 10, formatter: (v: number) => `${(v / 1000).toFixed(0)}k` },
      splitLine: { lineStyle: { color: '#1a1a25' } },
    },
    series: [{
      type: 'line',
      data: demandData.map(d => [d.timestamp, d.value]),
      smooth: true,
      symbol: 'none',
      lineStyle: { width: 1.5, color: '#3b82f6' },
      areaStyle: {
        color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [
          { offset: 0, color: 'rgba(59,130,246,0.2)' },
          { offset: 1, color: 'rgba(59,130,246,0.02)' },
        ]},
      },
    }],
    grid: { left: 55, right: 15, top: 30, bottom: 25 },
  };

  const baseloadOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis' as const,
      backgroundColor: '#1a1a25',
      borderColor: '#2a2a3a',
      textStyle: { color: '#e4e4e7', fontSize: 12 },
    },
    xAxis: {
      type: 'category' as const,
      data: baseloadData.map(d => d.date),
      axisLabel: { color: '#71717a', fontSize: 9, rotate: 45 },
      axisLine: { lineStyle: { color: '#2a2a3a' } },
    },
    yAxis: {
      type: 'value' as const,
      name: 'MW',
      nameTextStyle: { color: '#71717a', fontSize: 10 },
      axisLabel: { color: '#71717a', fontSize: 10, formatter: (v: number) => `${(v / 1000).toFixed(0)}k` },
      splitLine: { lineStyle: { color: '#1a1a25' } },
    },
    series: [{
      type: 'bar',
      data: baseloadData.map(d => d.base_load_mw),
      itemStyle: {
        color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [
          { offset: 0, color: '#22c55e' },
          { offset: 1, color: '#15803d' },
        ]},
        borderRadius: [3, 3, 0, 0],
      },
    }],
    grid: { left: 55, right: 15, top: 30, bottom: 60 },
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold">Power & Water</h2>
          <p className="text-xs text-zinc-500">Grid demand and nightly base-load trends</p>
        </div>
        <select
          value={region}
          onChange={e => setRegion(e.target.value)}
          className="bg-[#1a1a25] border border-[#2a2a3a] rounded-lg px-3 py-1.5 text-sm text-zinc-300 cursor-pointer focus:outline-none focus:ring-1 focus:ring-blue-500/50"
        >
          <option value="PJM">PJM (Data Center Alley)</option>
          <option value="ERCOT">ERCOT (Texas)</option>
        </select>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div>
          <h3 className="text-sm text-zinc-400 mb-2">Grid Demand — 7d</h3>
          <ReactEChartsCore option={demandOption} style={{ height: 260 }} />
        </div>
        <div>
          <h3 className="text-sm text-zinc-400 mb-2">Night Base Load (2-5 AM) — 30d</h3>
          <ReactEChartsCore option={baseloadOption} style={{ height: 260 }} />
        </div>
      </div>
    </div>
  );
}
