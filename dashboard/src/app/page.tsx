'use client';

import dynamic from 'next/dynamic';

const ScoreOverview = dynamic(() => import('@/components/ScoreOverview'), { ssr: false });
const PowerWaterPanel = dynamic(() => import('@/components/PowerWaterPanel'), { ssr: false });
const ComputeMarketPanel = dynamic(() => import('@/components/ComputeMarketPanel'), { ssr: false });
const TokenEconomicsPanel = dynamic(() => import('@/components/TokenEconomicsPanel'), { ssr: false });
const BandwidthPanel = dynamic(() => import('@/components/BandwidthPanel'), { ssr: false });
const OpsPanel = dynamic(() => import('@/components/OpsPanel'), { ssr: false });
const QuarterlyPanel = dynamic(() => import('@/components/QuarterlyPanel'), { ssr: false });
const AlertsPanel = dynamic(() => import('@/components/AlertsPanel'), { ssr: false });
const ConnectorStatusPanel = dynamic(() => import('@/components/ConnectorStatusPanel'), { ssr: false });

export default function Home() {
  return (
    <div className="space-y-6">
      {/* Panel 1: Pulse Overview */}
      <ScoreOverview />

      {/* Panel 2: Power & Water */}
      <PowerWaterPanel />

      {/* Panel 3+4: Compute + Token */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ComputeMarketPanel />
        <TokenEconomicsPanel />
      </div>

      {/* Panel 5: Bandwidth */}
      <BandwidthPanel />

      {/* Panel 6: Ops & Buildout */}
      <OpsPanel />

      {/* Panel 7: Quarterly Fundamentals */}
      <QuarterlyPanel />

      {/* Alerts + Connector Health */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <AlertsPanel />
        <ConnectorStatusPanel />
      </div>

      {/* Footer */}
      <div className="text-center text-xs text-zinc-600 py-4">
        AI Capex Efficiency Dashboard — Static demo with embedded data.
        All 7 panels operational. Connect real APIs via the full-stack backend for live data.
      </div>
    </div>
  );
}
