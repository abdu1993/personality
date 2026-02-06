'use client';

import { generateAlerts } from '@/lib/demo-data';

const SEVERITY_STYLES: Record<string, string> = {
  warning: 'border-l-yellow-500 bg-yellow-500/5',
  info: 'border-l-blue-500 bg-blue-500/5',
  critical: 'border-l-red-500 bg-red-500/5',
};

const SEVERITY_DOT: Record<string, string> = {
  warning: 'bg-yellow-500',
  info: 'bg-blue-500',
  critical: 'bg-red-500',
};

export default function AlertsPanel() {
  const events = generateAlerts();

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-1">Recent Alerts</h2>
      <p className="text-xs text-zinc-500 mb-4">Signal divergence notifications</p>
      <div className="space-y-2">
        {events.map(evt => (
          <div
            key={evt.id}
            className={`border-l-4 rounded-r-lg px-4 py-3 ${SEVERITY_STYLES[evt.severity] || SEVERITY_STYLES.info}`}
          >
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <span className={`w-1.5 h-1.5 rounded-full ${SEVERITY_DOT[evt.severity] || 'bg-blue-500'}`} />
                <span className="text-sm font-medium capitalize">
                  {evt.alert_type.replace(/_/g, ' ')}
                </span>
              </div>
              <span className="text-xs text-zinc-500">
                {new Date(evt.fired_at).toLocaleString()}
              </span>
            </div>
            <p className="text-xs text-zinc-400 pl-3.5">{evt.message}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
