# AI Capex Efficiency Dashboard

Tracks whether hyperscaler AI capex is translating into real, utilized compute versus idle shells. Ingests near-real-time "exhaust" signals across power, water/cooling, bandwidth, compute pricing, and market clearing to produce a composite **Utilization Confidence Score** (0–100).

## Architecture

```
┌─────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌────────────┐
│  Connectors │──▶│  TimescaleDB │◀──│  FastAPI Backend  │◀──│  Next.js   │
│  (Pollers)  │   │  (Postgres)  │   │  (REST API)       │   │  Frontend  │
└─────────────┘   └──────────────┘   └──────────────────┘   └────────────┘
      │                                       │
      ▼                                       ▼
   Scheduler                            Alert Engine
  (APScheduler)                     (Webhook / Email)
```

- **Ingestion**: Scheduled pollers via APScheduler + manual trigger endpoints
- **Storage**: PostgreSQL + TimescaleDB hypertables
- **Backend**: Python FastAPI serving pre-aggregated series
- **Frontend**: Next.js 14 + React + ECharts
- **Deployment**: Docker Compose

## Quick Start

```bash
# 1. Copy environment file and add your API keys
cp .env.example .env

# 2. Start all services
docker compose up --build

# 3. Access the dashboard
open http://localhost:3000

# API docs (Swagger)
open http://localhost:8000/docs
```

The system generates demo data automatically when API keys are not configured, so the dashboard works out of the box.

## Required API Keys

| Key | Source | Required for |
|-----|--------|-------------|
| `EIA_API_KEY` | [EIA Open Data](https://www.eia.gov/opendata/register.php) | Real grid demand data (PJM, ERCOT) |
| `CLOUDFLARE_API_TOKEN` | [Cloudflare Radar](https://developers.cloudflare.com/radar/) | HTTP traffic trend data |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | AWS Console | EC2 GPU spot price history |
| `ALERT_WEBHOOK_URL` | Slack/Discord webhook | Alert notifications |

All keys are optional — the system falls back to realistic demo data when keys are missing.

## Dashboard Panels

1. **Pulse Overview** — Composite Utilization Confidence Score (0–100) with per-component breakdown
2. **Power & Water** — Grid demand vs time, night base-load trend (PJM / ERCOT)
3. **Compute Market Tightness** — GPU spot price history by GPU type (H100, A100, A10G, L4)
4. **Token Economics** — Input/output $/1M tokens by provider/model, deflation rate
5. **Bandwidth / Inference Exhaust** — HTTP traffic trends (Cloudflare Radar proxy)
6. **Alerts** — Fired alert events with severity indicators
7. **Connector Health** — Last run status, row counts, errors

## MVP Connectors (Implemented)

| Connector | Source | Cadence | Metric IDs |
|-----------|--------|---------|------------|
| `eia_grid` | EIA-930 Hourly Grid Monitor | Hourly | `grid.demand_mw` |
| `cloudflare_radar` | Cloudflare Radar API | 15 min | `net.http_requests`, `net.traffic_anomalies` |
| `token_pricing` | Manual pricing table | Daily | `model_price.input_usd_per_1m_tokens`, `model_price.output_usd_per_1m_tokens` |
| `gpu_spot` | AWS EC2 Spot API | 30 min | `gpu_spot.price_usd_per_hour` |

## Alert Rules

| Alert | Trigger |
|-------|---------|
| Overbuild Signal | GPU spot prices declining for 14d |
| Tightness Signal | GPU spot prices rising for 14d |
| Inference Ramp | Bandwidth growing faster than power demand |
| Capex–Power Divergence | Capex up but base load flat (requires manual capex input) |
| Shell Risk | Power up but labor ops postings flat (v2) |

## Composite Score Weights

Adjustable via the API (`PUT /api/score/weights`) or future UI controls:

| Component | Default Weight |
|-----------|---------------|
| Power / Base Load | 25 |
| GPU Spot Tightness | 15 |
| Token Deflation | 15 |
| Bandwidth Proxy | 10 |
| Water Anomaly | 10 |
| Labor Ops Ramp | 10 |
| Shadow Price Spread | 10 |
| PUE / Cooling | 5 |

## Adding New Connectors

1. Create a new file in `backend/app/connectors/` inheriting from `BaseConnector`
2. Implement the `fetch()` method returning `List[DataPoint]`
3. Register in `backend/app/connectors/__init__.py`
4. Add a scheduler job in `backend/app/scheduler.py`
5. Add to the manual trigger map in `backend/app/api/routes.py`

```python
from app.connectors.base import BaseConnector, DataPoint

class MyConnector(BaseConnector):
    name = "my_connector"
    schema_version = "1.0"

    async def fetch(self) -> list[DataPoint]:
        # Fetch, normalize, return DataPoints
        ...
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/metrics/series` | Fetch raw time series |
| GET | `/api/metrics/latest` | Latest value per region/provider |
| GET | `/api/metrics/available` | List all metric IDs |
| GET | `/api/derived/baseload` | Night base-load calculation |
| GET | `/api/derived/token-deflation` | Token deflation rates |
| GET | `/api/score` | Composite utilization score |
| GET/PUT | `/api/score/weights` | View/update score weights |
| GET | `/api/alerts/events` | Recent alert events |
| POST | `/api/alerts/evaluate` | Trigger alert evaluation |
| GET | `/api/connectors/status` | Connector health status |
| POST | `/api/connectors/run/{name}` | Manually run a connector |

## V2 Roadmap

- Water usage / cooling data connectors
- Labor/job postings connector (ops hiring signal)
- Secondary GPU market pricing (shadow clearing price)
- PUE document ingestion from sustainability reports
- Cooling chemicals/consumables index
- IX utilization tracking
- GPU failure/RMA velocity proxy
- Capex-to-revenue conversion tracking
- Silicon sovereignty index
- Weight adjustment UI controls
- Historical score snapshots with 7/30/90d change indicators
