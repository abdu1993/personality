'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { ConnectorStatus } from '@/types';

export default function ConnectorStatusPanel() {
  const [connectors, setConnectors] = useState<ConnectorStatus[]>([]);

  useEffect(() => {
    api.getConnectorStatus()
      .then(d => setConnectors(d.connectors || []))
      .catch(() => {});
  }, []);

  const statusColor = (s: string) => {
    switch (s) {
      case 'success': return 'text-green-400';
      case 'error': return 'text-red-400';
      case 'rate_limited': return 'text-yellow-400';
      default: return 'text-zinc-400';
    }
  };

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4">Connector Health</h2>
      {connectors.length === 0 ? (
        <p className="text-sm text-zinc-500">No connector runs recorded yet.</p>
      ) : (
        <div className="space-y-2">
          {connectors.map(c => (
            <div key={c.name} className="flex items-center justify-between bg-[#1a1a25] rounded px-3 py-2">
              <div>
                <span className="text-sm font-medium">{c.name}</span>
                <span className={`ml-2 text-xs ${statusColor(c.status)}`}>{c.status}</span>
              </div>
              <div className="text-right">
                <span className="text-xs text-zinc-500">
                  {c.last_run ? new Date(c.last_run).toLocaleString() : 'Never'}
                </span>
                <span className="text-xs text-zinc-400 ml-2">{c.rows_ingested} rows</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
