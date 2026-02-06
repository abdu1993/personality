'use client';

import dynamic from 'next/dynamic';

// Dynamic imports to avoid SSR issues with ECharts
const ScoreOverview = dynamic(() => import('@/components/ScoreOverview'), { ssr: false });
const PowerWaterPanel = dynamic(() => import('@/components/PowerWaterPanel'), { ssr: false });
const ComputeMarketPanel = dynamic(() => import('@/components/ComputeMarketPanel'), { ssr: false });
const TokenEconomicsPanel = dynamic(() => import('@/components/TokenEconomicsPanel'), { ssr: false });
const BandwidthPanel = dynamic(() => import('@/components/BandwidthPanel'), { ssr: false });
const AlertsPanel = dynamic(() => import('@/components/AlertsPanel'), { ssr: false });
const ConnectorStatusPanel = dynamic(() => import('@/components/ConnectorStatusPanel'), { ssr: false });

export default function Home() {
  return (
    <div className="space-y-6">
      {/* Panel 1: Pulse Overview */}
      <ScoreOverview />

      {/* Panel 2: Power & Water */}
      <PowerWaterPanel />

      {/* Row: Compute + Token */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Panel 3: Compute Market Tightness */}
        <ComputeMarketPanel />
        {/* Panel 4: Token Economics */}
        <TokenEconomicsPanel />
      </div>

      {/* Panel 5: Bandwidth */}
      <BandwidthPanel />

      {/* Row: Alerts + Connector Health */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <AlertsPanel />
        <ConnectorStatusPanel />
      </div>
    </div>
  );
}
