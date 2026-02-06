"""Composite Utilization Confidence Score (0-100).

Aggregates derived metrics into a single score with transparent,
adjustable weights.
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np


async def get_weights(session: AsyncSession) -> Dict[str, float]:
    """Load current weights from DB."""
    result = await session.execute(text("SELECT component, weight FROM score_weights"))
    rows = result.fetchall()
    return {r[0]: r[1] for r in rows}


async def _query_recent(
    session: AsyncSession,
    metric_id: str,
    days: int = 7,
    region: Optional[str] = None,
) -> List[float]:
    """Get recent metric values."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    q = text("""
        SELECT value FROM metric_data
        WHERE metric_id = :mid AND timestamp_utc >= :cutoff
    """ + (" AND region = :region" if region else "") + """
        ORDER BY timestamp_utc DESC
    """)
    params: Dict[str, Any] = {"mid": metric_id, "cutoff": cutoff}
    if region:
        params["region"] = region
    result = await session.execute(q, params)
    return [float(r[0]) for r in result.fetchall()]


def _zscore_clip(values: List[float], higher_is_more_utilized: bool = True) -> float:
    """Convert values to a 0-1 score via z-score clipping.
    Returns latest value's percentile-like score."""
    if not values or len(values) < 2:
        return 0.5
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-9:
        return 0.5
    z = (arr[0] - mean) / std  # latest value
    # Clip to [-3, 3] then normalize to [0, 1]
    z_clipped = np.clip(z, -3, 3)
    score = (z_clipped + 3) / 6
    if not higher_is_more_utilized:
        score = 1 - score
    return float(score)


async def compute_score(session: AsyncSession) -> Dict[str, Any]:
    """Compute the composite Utilization Confidence Score."""
    weights = await get_weights(session)

    components: Dict[str, Dict[str, Any]] = {}

    # 1. Power base load (higher base load = more utilization)
    power_vals = await _query_recent(session, "grid.demand_mw", days=30, region="PJM")
    power_score = _zscore_clip(power_vals, higher_is_more_utilized=True)
    components["power_baseload"] = {
        "score": power_score,
        "weight": weights.get("power_baseload", 25),
        "data_points": len(power_vals),
        "confidence": 0.9 if power_vals else 0.0,
    }

    # 2. GPU spot tightness (higher price = more utilization)
    gpu_vals = await _query_recent(session, "gpu_spot.price_usd_per_hour", days=14)
    gpu_score = _zscore_clip(gpu_vals, higher_is_more_utilized=True)
    components["gpu_spot_tightness"] = {
        "score": gpu_score,
        "weight": weights.get("gpu_spot_tightness", 15),
        "data_points": len(gpu_vals),
        "confidence": 0.85 if gpu_vals else 0.0,
    }

    # 3. Token deflation (lower prices = more efficiency/utilization)
    token_vals = await _query_recent(session, "model_price.input_usd_per_1m_tokens", days=30)
    token_score = _zscore_clip(token_vals, higher_is_more_utilized=False)
    components["token_deflation"] = {
        "score": token_score,
        "weight": weights.get("token_deflation", 15),
        "data_points": len(token_vals),
        "confidence": 0.85 if token_vals else 0.0,
    }

    # 4. Bandwidth proxy (higher traffic = more inference)
    bw_vals = await _query_recent(session, "net.http_requests", days=7)
    bw_score = _zscore_clip(bw_vals, higher_is_more_utilized=True)
    components["bandwidth_proxy"] = {
        "score": bw_score,
        "weight": weights.get("bandwidth_proxy", 10),
        "data_points": len(bw_vals),
        "confidence": 0.7 if bw_vals else 0.0,
    }

    # 5-8. Placeholder components (v2 — water, labor, shadow price, PUE)
    for placeholder in ["water_anomaly", "labor_ops_ramp", "shadow_price_spread", "pue_cooling"]:
        components[placeholder] = {
            "score": 0.5,  # neutral
            "weight": weights.get(placeholder, 5),
            "data_points": 0,
            "confidence": 0.0,
        }

    # Compute weighted composite
    total_weighted = 0.0
    total_weight = 0.0
    for name, comp in components.items():
        w = comp["weight"]
        conf = comp["confidence"]
        effective_weight = w * conf
        total_weighted += comp["score"] * effective_weight
        total_weight += effective_weight

    composite = (total_weighted / total_weight * 100) if total_weight > 0 else 50.0
    composite = round(min(100, max(0, composite)), 1)

    return {
        "utilization_confidence": composite,
        "components": components,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
