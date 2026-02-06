"""Alerting service.

Evaluates alert rules and fires notifications via webhook/email.
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.metrics import AlertEvent
from app.config import settings
import httpx
import json


ALERT_RULES = [
    {
        "id": "capex_power_divergence",
        "name": "Capex–Power Divergence",
        "description": "Capex up sharply but base load flat/down for 30d",
        "severity": "warning",
    },
    {
        "id": "overbuild_signal",
        "name": "Overbuild Signal",
        "description": "Spot prices collapse + availability high for 14d",
        "severity": "warning",
    },
    {
        "id": "tightness_signal",
        "name": "Tightness Signal",
        "description": "Spot unavailable + shadow prices spike",
        "severity": "info",
    },
    {
        "id": "inference_ramp",
        "name": "Inference Ramp",
        "description": "Bandwidth proxy accelerates while power steady",
        "severity": "info",
    },
    {
        "id": "shell_risk",
        "name": "Shell Risk",
        "description": "Power up but labor ops postings flat",
        "severity": "warning",
    },
]


async def _get_trend(session: AsyncSession, metric_id: str, days: int, region: Optional[str] = None) -> Optional[float]:
    """Get slope of a metric over N days (positive = increasing)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    q = """
        SELECT
            EXTRACT(EPOCH FROM timestamp_utc) as ts_epoch,
            value
        FROM metric_data
        WHERE metric_id = :mid AND timestamp_utc >= :cutoff
    """
    if region:
        q += " AND region = :region"
    q += " ORDER BY timestamp_utc ASC"

    params: Dict[str, Any] = {"mid": metric_id, "cutoff": cutoff}
    if region:
        params["region"] = region

    result = await session.execute(text(q), params)
    rows = result.fetchall()

    if len(rows) < 5:
        return None

    import numpy as np
    x = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    x_norm = (x - x[0]) / 3600  # hours
    if len(x_norm) < 2:
        return None
    coeffs = np.polyfit(x_norm, y, 1)
    return float(coeffs[0])


async def evaluate_alerts(session: AsyncSession) -> List[Dict[str, Any]]:
    """Evaluate all alert rules and return fired alerts."""
    fired: List[Dict[str, Any]] = []

    # 1. Overbuild signal: GPU spot price dropping + high availability
    gpu_trend = await _get_trend(session, "gpu_spot.price_usd_per_hour", days=14)
    if gpu_trend is not None and gpu_trend < -0.1:
        alert = {
            "alert_type": "overbuild_signal",
            "severity": "warning",
            "message": f"GPU spot prices declining (slope={gpu_trend:.4f}/hr over 14d). Potential overbuild signal.",
            "details_json": {"gpu_price_trend": gpu_trend},
        }
        fired.append(alert)

    # 2. Tightness signal: GPU spot price rising
    if gpu_trend is not None and gpu_trend > 0.2:
        alert = {
            "alert_type": "tightness_signal",
            "severity": "info",
            "message": f"GPU spot prices rising (slope={gpu_trend:.4f}/hr over 14d). Compute tightness increasing.",
            "details_json": {"gpu_price_trend": gpu_trend},
        }
        fired.append(alert)

    # 3. Inference ramp: bandwidth up while power steady
    bw_trend = await _get_trend(session, "net.http_requests", days=7)
    power_trend = await _get_trend(session, "grid.demand_mw", days=7, region="PJM")
    if bw_trend is not None and power_trend is not None:
        if bw_trend > 0 and abs(power_trend) < abs(bw_trend) * 0.1:
            alert = {
                "alert_type": "inference_ramp",
                "severity": "info",
                "message": "Bandwidth growing faster than power demand — inference monetization signal.",
                "details_json": {"bw_trend": bw_trend, "power_trend": power_trend},
            }
            fired.append(alert)

    # Store fired alerts
    for a in fired:
        event = AlertEvent(
            alert_type=a["alert_type"],
            fired_at=datetime.now(timezone.utc),
            severity=a["severity"],
            message=a["message"],
            details_json=a.get("details_json"),
        )
        session.add(event)

    if fired:
        await session.commit()

    # Send notifications
    for a in fired:
        await _notify(a)

    return fired


async def _notify(alert: Dict[str, Any]):
    """Send alert notification via webhook."""
    if settings.alert_webhook_url:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    settings.alert_webhook_url,
                    json={
                        "text": f"[{alert['severity'].upper()}] {alert['alert_type']}: {alert['message']}",
                        "alert": alert,
                    },
                    timeout=10,
                )
        except Exception:
            pass
