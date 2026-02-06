'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { AlertEvent } from '@/types';

const SEVERITY_STYLES: Record<string, string> = {
  warning: 'border-l-yellow-500 bg-yellow-500/5',
  info: 'border-l-blue-500 bg-blue-500/5',
  critical: 'border-l-red-500 bg-red-500/5',
};

export default function AlertsPanel() {
  const [events, setEvents] = useState<AlertEvent[]>([]);

  useEffect(() => {
    api.getAlertEvents(30)
      .then(d => setEvents(d.events || []))
      .catch(() => {});
  }, []);

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4">Recent Alerts</h2>
      {events.length === 0 ? (
        <p className="text-sm text-zinc-500">No alerts fired in the last 30 days.</p>
      ) : (
        <div className="space-y-2 max-h-[300px] overflow-y-auto">
          {events.map(evt => (
            <div
              key={evt.id}
              className={`border-l-4 rounded-r px-3 py-2 ${SEVERITY_STYLES[evt.severity] || SEVERITY_STYLES.info}`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{evt.alert_type.replace(/_/g, ' ')}</span>
                <span className="text-xs text-zinc-500">
                  {new Date(evt.fired_at).toLocaleString()}
                </span>
              </div>
              <p className="text-xs text-zinc-400 mt-1">{evt.message}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
