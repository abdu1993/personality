"""APScheduler-based scheduler for periodic connector runs and alert evaluation."""
import asyncio
from datetime import datetime, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.config import settings
from app.connectors.eia import EIAConnector
from app.connectors.cloudflare_radar import CloudflareRadarConnector
from app.connectors.token_pricing import TokenPricingConnector
from app.connectors.gpu_spot import GPUSpotConnector
from app.services.ingestion import run_connector
from app.services.alerts import evaluate_alerts
from app.migrations.run import run_migrations

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def run_eia():
    async with async_session() as session:
        connector = EIAConnector()
        count = await run_connector(connector, session)
        print(f"[{datetime.now(timezone.utc)}] EIA: {count} rows ingested")


async def run_cloudflare():
    async with async_session() as session:
        connector = CloudflareRadarConnector()
        count = await run_connector(connector, session)
        print(f"[{datetime.now(timezone.utc)}] Cloudflare: {count} rows ingested")


async def run_token_pricing():
    async with async_session() as session:
        connector = TokenPricingConnector()
        count = await run_connector(connector, session)
        print(f"[{datetime.now(timezone.utc)}] Token Pricing: {count} rows ingested")


async def run_gpu_spot():
    async with async_session() as session:
        connector = GPUSpotConnector()
        count = await run_connector(connector, session)
        print(f"[{datetime.now(timezone.utc)}] GPU Spot: {count} rows ingested")


async def run_alerts():
    async with async_session() as session:
        fired = await evaluate_alerts(session)
        print(f"[{datetime.now(timezone.utc)}] Alerts evaluated: {len(fired)} fired")


def main():
    # Run migrations first
    run_migrations()

    scheduler = AsyncIOScheduler()

    # EIA: hourly
    scheduler.add_job(run_eia, "interval", hours=1, id="eia",
                      next_run_time=datetime.now(timezone.utc))

    # Cloudflare Radar: every 15 minutes
    scheduler.add_job(run_cloudflare, "interval", minutes=15, id="cloudflare",
                      next_run_time=datetime.now(timezone.utc))

    # Token pricing: daily
    scheduler.add_job(run_token_pricing, "interval", hours=24, id="token_pricing",
                      next_run_time=datetime.now(timezone.utc))

    # GPU spot: every 30 minutes
    scheduler.add_job(run_gpu_spot, "interval", minutes=30, id="gpu_spot",
                      next_run_time=datetime.now(timezone.utc))

    # Alert evaluation: every 15 minutes
    scheduler.add_job(run_alerts, "interval", minutes=15, id="alerts")

    scheduler.start()
    print("Scheduler started. Press Ctrl+C to exit.")

    loop = asyncio.get_event_loop()
    try:
        loop.run_forever()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()


if __name__ == "__main__":
    main()
