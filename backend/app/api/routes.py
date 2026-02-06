"""REST API endpoints for the AI Capex Dashboard."""
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.scoring import compute_score, get_weights
from app.services.alerts import evaluate_alerts, ALERT_RULES
from app.services.ingestion import run_connector
from app.connectors import (
    EIAConnector, CloudflareRadarConnector, TokenPricingConnector, GPUSpotConnector
)

router = APIRouter(prefix="/api")


# ── Time Series ──────────────────────────────────────────────────────────────

@router.get("/metrics/series")
async def get_metric_series(
    metric_id: str = Query(..., description="e.g. grid.demand_mw"),
    region: Optional[str] = Query(None),
    provider: Optional[str] = Query(None),
    days: int = Query(7, ge=1, le=365),
    limit: int = Query(2000, ge=1, le=10000),
    db: AsyncSession = Depends(get_db),
):
    """Fetch raw time series for a given metric."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    q = """
        SELECT timestamp_utc, value, unit, region, source, provider, confidence, metadata_json
        FROM metric_data
        WHERE metric_id = :mid AND timestamp_utc >= :cutoff
    """
    params: Dict[str, Any] = {"mid": metric_id, "cutoff": cutoff}
    if region:
        q += " AND region = :region"
        params["region"] = region
    if provider:
        q += " AND provider = :provider"
        params["provider"] = provider
    q += " ORDER BY timestamp_utc ASC LIMIT :lim"
    params["lim"] = limit

    result = await db.execute(text(q), params)
    rows = result.fetchall()

    return {
        "metric_id": metric_id,
        "count": len(rows),
        "data": [
            {
                "timestamp": r[0].isoformat() if r[0] else None,
                "value": r[1],
                "unit": r[2],
                "region": r[3],
                "source": r[4],
                "provider": r[5],
                "confidence": r[6],
                "metadata": r[7],
            }
            for r in rows
        ],
    }


@router.get("/metrics/latest")
async def get_latest_metrics(
    metric_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Get the most recent value for each region/provider combo."""
    q = text("""
        SELECT DISTINCT ON (region, provider)
            timestamp_utc, value, unit, region, provider, confidence, metadata_json
        FROM metric_data
        WHERE metric_id = :mid
        ORDER BY region, provider, timestamp_utc DESC
    """)
    result = await db.execute(q, {"mid": metric_id})
    rows = result.fetchall()

    return {
        "metric_id": metric_id,
        "data": [
            {
                "timestamp": r[0].isoformat() if r[0] else None,
                "value": r[1],
                "unit": r[2],
                "region": r[3],
                "provider": r[4],
                "confidence": r[5],
                "metadata": r[6],
            }
            for r in rows
        ],
    }


@router.get("/metrics/available")
async def list_available_metrics(db: AsyncSession = Depends(get_db)):
    """List all available metric_ids with counts."""
    result = await db.execute(text("""
        SELECT metric_id, COUNT(*) as cnt,
               MIN(timestamp_utc) as first_ts,
               MAX(timestamp_utc) as last_ts
        FROM metric_data
        GROUP BY metric_id
        ORDER BY metric_id
    """))
    rows = result.fetchall()
    return {
        "metrics": [
            {
                "metric_id": r[0],
                "count": r[1],
                "first_timestamp": r[2].isoformat() if r[2] else None,
                "last_timestamp": r[3].isoformat() if r[3] else None,
            }
            for r in rows
        ]
    }


# ── Derived / Night Base Load ────────────────────────────────────────────────

@router.get("/derived/baseload")
async def get_baseload(
    region: str = Query("PJM"),
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Compute nightly base load (median demand 2-5 AM) per day."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db.execute(text("""
        SELECT
            DATE(timestamp_utc) as day,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as median_load
        FROM metric_data
        WHERE metric_id = 'grid.demand_mw'
          AND region = :region
          AND timestamp_utc >= :cutoff
          AND EXTRACT(HOUR FROM timestamp_utc) BETWEEN 2 AND 4
        GROUP BY DATE(timestamp_utc)
        ORDER BY day ASC
    """), {"region": region, "cutoff": cutoff})
    rows = result.fetchall()

    return {
        "region": region,
        "data": [
            {"date": r[0].isoformat() if r[0] else None, "base_load_mw": round(r[1], 1)}
            for r in rows
        ],
    }


# ── Composite Score ──────────────────────────────────────────────────────────

@router.get("/score")
async def get_utilization_score(db: AsyncSession = Depends(get_db)):
    """Get the composite Utilization Confidence Score."""
    return await compute_score(db)


@router.get("/score/weights")
async def get_score_weights(db: AsyncSession = Depends(get_db)):
    """Get current scoring weights."""
    return await get_weights(db)


@router.put("/score/weights")
async def update_score_weights(
    weights: Dict[str, float],
    db: AsyncSession = Depends(get_db),
):
    """Update scoring weights."""
    for component, weight in weights.items():
        await db.execute(
            text("UPDATE score_weights SET weight = :w, updated_at = NOW() WHERE component = :c"),
            {"w": weight, "c": component},
        )
    await db.commit()
    return await get_weights(db)


# ── Alerts ───────────────────────────────────────────────────────────────────

@router.get("/alerts/rules")
async def list_alert_rules():
    """List configured alert rules."""
    return {"rules": ALERT_RULES}


@router.get("/alerts/events")
async def list_alert_events(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """List recent alert events."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db.execute(text("""
        SELECT id, alert_type, fired_at, severity, message, details_json, acknowledged
        FROM alert_events
        WHERE fired_at >= :cutoff
        ORDER BY fired_at DESC
        LIMIT 100
    """), {"cutoff": cutoff})
    rows = result.fetchall()

    return {
        "events": [
            {
                "id": r[0],
                "alert_type": r[1],
                "fired_at": r[2].isoformat() if r[2] else None,
                "severity": r[3],
                "message": r[4],
                "details": r[5],
                "acknowledged": bool(r[6]),
            }
            for r in rows
        ]
    }


@router.post("/alerts/evaluate")
async def trigger_alert_evaluation(db: AsyncSession = Depends(get_db)):
    """Manually trigger alert evaluation."""
    fired = await evaluate_alerts(db)
    return {"fired": fired}


# ── Connector Health ─────────────────────────────────────────────────────────

@router.get("/connectors/status")
async def connector_status(db: AsyncSession = Depends(get_db)):
    """Get last run status of each connector."""
    result = await db.execute(text("""
        SELECT DISTINCT ON (connector_name)
            connector_name, run_at, status, rows_ingested, error_message
        FROM connector_log
        ORDER BY connector_name, run_at DESC
    """))
    rows = result.fetchall()

    return {
        "connectors": [
            {
                "name": r[0],
                "last_run": r[1].isoformat() if r[1] else None,
                "status": r[2],
                "rows_ingested": r[3],
                "error": r[4],
            }
            for r in rows
        ]
    }


@router.post("/connectors/run/{connector_name}")
async def run_connector_manual(connector_name: str, db: AsyncSession = Depends(get_db)):
    """Manually trigger a connector run."""
    connectors = {
        "eia_grid": EIAConnector,
        "cloudflare_radar": CloudflareRadarConnector,
        "token_pricing": TokenPricingConnector,
        "gpu_spot": GPUSpotConnector,
    }
    cls = connectors.get(connector_name)
    if not cls:
        raise HTTPException(404, f"Unknown connector: {connector_name}")

    connector = cls()
    count = await run_connector(connector, db)
    return {"connector": connector_name, "rows_ingested": count}


# ── Token pricing helpers ────────────────────────────────────────────────────

@router.get("/derived/token-deflation")
async def get_token_deflation(
    days: int = Query(90, ge=7, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Get token deflation rates by provider."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db.execute(text("""
        SELECT provider, metadata_json->>'model' as model,
               MIN(value) as min_price, MAX(value) as max_price,
               (array_agg(value ORDER BY timestamp_utc DESC))[1] as latest,
               (array_agg(value ORDER BY timestamp_utc ASC))[1] as earliest
        FROM metric_data
        WHERE metric_id = 'model_price.input_usd_per_1m_tokens'
          AND timestamp_utc >= :cutoff
        GROUP BY provider, metadata_json->>'model'
        ORDER BY provider, model
    """), {"cutoff": cutoff})
    rows = result.fetchall()

    data = []
    for r in rows:
        earliest = r[5] if r[5] else 0
        latest = r[4] if r[4] else 0
        deflation_pct = ((latest - earliest) / earliest * 100) if earliest > 0 else 0
        data.append({
            "provider": r[0],
            "model": r[1],
            "latest_price": latest,
            "earliest_price": earliest,
            "deflation_pct": round(deflation_pct, 2),
        })

    return {"data": data}
