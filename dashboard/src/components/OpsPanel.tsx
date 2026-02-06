'use client';

import { useMemo } from 'react';
import ReactEChartsCore from 'echarts-for-react';
import { generateLaborData } from '@/lib/demo-data';

export default function OpsPanel() {
  const data = useMemo(() => generateLaborData(60), []);

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis' as const,
      backgroundColor: '#1a1a25',
      borderColor: '#2a2a3a',
      textStyle: { color: '#e4e4e7', fontSize: 12 },
    },
    legend: {
      data: ['Ops Roles', 'Construction', 'Cooling Specialists'],
      textStyle: { color: '#a1a1aa', fontSize: 11 },
      top: 0,
      icon: 'roundRect',
      itemWidth: 14,
      itemHeight: 3,
    },
    xAxis: {
      type: 'category' as const,
      data: data.map(d => d.date),
      axisLabel: { color: '#71717a', fontSize: 9, rotate: 45, interval: 6 },
      axisLine: { lineStyle: { color: '#2a2a3a' } },
    },
    yAxis: {
      type: 'value' as const,
      name: 'Postings',
      nameTextStyle: { color: '#71717a', fontSize: 10 },
      axisLabel: { color: '#71717a', fontSize: 10 },
      splitLine: { lineStyle: { color: '#1a1a25' } },
    },
    series: [
      {
        name: 'Ops Roles',
        type: 'line',
        data: data.map(d => d.ops_roles),
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 1.5, color: '#3b82f6' },
      },
      {
        name: 'Construction',
        type: 'line',
        data: data.map(d => d.construction_roles),
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 1.5, color: '#71717a' },
      },
      {
        name: 'Cooling Specialists',
        type: 'line',
        data: data.map(d => d.cooling_roles),
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 1.5, color: '#06b6d4' },
      },
    ],
    grid: { left: 45, right: 15, top: 35, bottom: 60 },
  };

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-1">Ops & Buildout</h2>
      <p className="text-xs text-zinc-500 mb-4">Job postings trend — ops vs construction vs cooling</p>
      <ReactEChartsCore option={option} style={{ height: 280 }} />
    </div>
  );
}
