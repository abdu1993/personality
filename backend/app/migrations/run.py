"""Run database migrations: create tables and TimescaleDB hypertables."""
import sqlalchemy
from sqlalchemy import text
from app.config import settings
from app.database import Base
from app.models import MetricData, ConnectorLog, AlertEvent, ScoreWeights


DEFAULT_WEIGHTS = {
    "power_baseload": 25,
    "gpu_spot_tightness": 15,
    "token_deflation": 15,
    "bandwidth_proxy": 10,
    "water_anomaly": 10,
    "labor_ops_ramp": 10,
    "shadow_price_spread": 10,
    "pue_cooling": 5,
}


def run_migrations():
    engine = sqlalchemy.create_engine(settings.database_url_sync)

    with engine.begin() as conn:
        # Enable TimescaleDB extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))

    # Create all tables
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        # Convert metric_data to a hypertable if not already
        try:
            conn.execute(text(
                "SELECT create_hypertable('metric_data', 'timestamp_utc', "
                "if_not_exists => TRUE, migrate_data => TRUE);"
            ))
        except Exception:
            pass  # Already a hypertable

        # Seed default weights
        for component, weight in DEFAULT_WEIGHTS.items():
            conn.execute(text(
                "INSERT INTO score_weights (component, weight, updated_at) "
                "VALUES (:comp, :w, NOW()) "
                "ON CONFLICT (component) DO NOTHING"
            ), {"comp": component, "w": weight})

    engine.dispose()
    print("Migrations complete.")


if __name__ == "__main__":
    run_migrations()
