"""Ingestion service: runs connectors and stores data."""
from datetime import datetime, timezone
from typing import List, Type
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.connectors.base import BaseConnector, DataPoint
from app.models.metrics import MetricData, ConnectorLog


async def run_connector(connector: BaseConnector, session: AsyncSession) -> int:
    """Run a single connector: fetch, store, log."""
    log_entry = ConnectorLog(
        connector_name=connector.name,
        run_at=datetime.now(timezone.utc),
        status="running",
        rows_ingested=0,
        schema_version=connector.schema_version,
    )

    try:
        points = await connector.fetch()
        count = 0

        if points:
            # Bulk insert via raw SQL for performance
            values = []
            for p in points:
                values.append({
                    "timestamp_utc": p.timestamp_utc,
                    "metric_id": p.metric_id,
                    "value": p.value,
                    "unit": p.unit,
                    "region": p.region,
                    "source": p.source,
                    "provider": p.provider,
                    "confidence": p.confidence,
                    "metadata_json": p.metadata_json,
                })

            # Insert in batches of 500
            batch_size = 500
            for i in range(0, len(values), batch_size):
                batch = values[i:i + batch_size]
                stmt = text("""
                    INSERT INTO metric_data
                    (timestamp_utc, metric_id, value, unit, region, source, provider, confidence, metadata_json)
                    VALUES
                    (:timestamp_utc, :metric_id, :value, :unit, :region, :source, :provider, :confidence, CAST(:metadata_json AS jsonb))
                """)
                for row in batch:
                    # Convert metadata_json to string for the cast
                    import json
                    md = json.dumps(row["metadata_json"]) if row["metadata_json"] else None
                    row["metadata_json"] = md
                    await session.execute(stmt, row)
                    count += 1

        await session.commit()

        log_entry.status = "success"
        log_entry.rows_ingested = count

    except Exception as e:
        log_entry.status = "error"
        log_entry.error_message = str(e)[:1000]
        await session.rollback()

    finally:
        # Log the connector run
        session.add(log_entry)
        await session.commit()
        await connector.close()

    return log_entry.rows_ingested
