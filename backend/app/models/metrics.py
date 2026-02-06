from sqlalchemy import (
    Column, String, Float, DateTime, Integer, Text, JSON, Index,
    BigInteger, func
)
from app.database import Base
from datetime import datetime


class MetricData(Base):
    """Core time-series table for all metric datapoints."""
    __tablename__ = "metric_data"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp_utc = Column(DateTime(timezone=True), nullable=False)
    metric_id = Column(String(128), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(64), nullable=False)
    region = Column(String(128), nullable=False, default="global")
    source = Column(String(128), nullable=False)
    provider = Column(String(128), nullable=False, default="market-wide")
    confidence = Column(Float, nullable=False, default=0.5)
    metadata_json = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_metric_ts", "metric_id", "timestamp_utc"),
        Index("idx_metric_provider", "metric_id", "provider"),
        Index("idx_metric_region", "metric_id", "region"),
    )


class ConnectorLog(Base):
    """Tracks connector health and ingestion runs."""
    __tablename__ = "connector_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    connector_name = Column(String(128), nullable=False)
    run_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    status = Column(String(32), nullable=False)  # success, error, rate_limited
    rows_ingested = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    schema_version = Column(String(32), nullable=True)
    raw_response_path = Column(String(512), nullable=True)


class AlertEvent(Base):
    """Stores fired alerts."""
    __tablename__ = "alert_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    alert_type = Column(String(128), nullable=False)
    fired_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    severity = Column(String(32), nullable=False, default="warning")
    message = Column(Text, nullable=False)
    details_json = Column(JSON, nullable=True)
    acknowledged = Column(Integer, nullable=False, default=0)


class ScoreWeights(Base):
    """Adjustable weights for the composite utilization score."""
    __tablename__ = "score_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    component = Column(String(128), nullable=False, unique=True)
    weight = Column(Float, nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
