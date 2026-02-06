'use client';

import { getConnectorStatus } from '@/lib/demo-data';

const CONNECTOR_LABELS: Record<string, string> = {
  eia_grid: 'EIA Grid Monitor',
  cloudflare_radar: 'Cloudflare Radar',
  token_pricing: 'Token Pricing',
  gpu_spot: 'GPU Spot Pricing',
};

export default function ConnectorStatusPanel() {
  const connectors = getConnectorStatus();

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-1">Connector Health</h2>
      <p className="text-xs text-zinc-500 mb-4">Ingestion pipeline status</p>
      <div className="space-y-2">
        {connectors.map(c => {
          const ago = Math.round((Date.now() - new Date(c.last_run).getTime()) / 60000);
          return (
            <div key={c.name} className="inner-card flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <span className="w-2 h-2 rounded-full bg-green-500" />
                <div>
                  <span className="text-sm font-medium">{CONNECTOR_LABELS[c.name] || c.name}</span>
                  <span className="text-xs text-zinc-600 ml-2">{c.rows_ingested} rows</span>
                </div>
              </div>
              <span className="text-xs text-zinc-500">{ago}m ago</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
