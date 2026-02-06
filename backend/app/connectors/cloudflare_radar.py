"""Cloudflare Radar connector for bandwidth/traffic trends.

Uses the Cloudflare Radar API to fetch HTTP request trends
as a proxy for inference-related egress traffic.
"""
from datetime import datetime, timezone, timedelta
from typing import List
from app.connectors.base import BaseConnector, DataPoint
from app.config import settings


class CloudflareRadarConnector(BaseConnector):
    name = "cloudflare_radar"
    schema_version = "1.0"

    BASE_URL = "https://api.cloudflare.com/client/v4/radar"

    async def fetch(self) -> List[DataPoint]:
        if not settings.cloudflare_api_token:
            return self._generate_demo_data()

        points: List[DataPoint] = []
        headers = {
            "Authorization": f"Bearer {settings.cloudflare_api_token}",
        }

        # Fetch HTTP requests timeseries for US
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=7)

        try:
            resp = await self.client.get(
                f"{self.BASE_URL}/http/timeseries",
                headers=headers,
                params={
                    "dateStart": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "dateEnd": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "location": "US",
                    "aggInterval": "1h",
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            await self.store_raw(data, "http_timeseries_US")

            series = (
                data.get("result", {})
                .get("http_requests", {})
                .get("timestamps", [])
            )
            values = (
                data.get("result", {})
                .get("http_requests", {})
                .get("values", [])
            )

            for ts_str, val in zip(series, values):
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
                points.append(DataPoint(
                    timestamp_utc=ts,
                    metric_id="net.http_requests",
                    value=float(val),
                    unit="requests",
                    region="US",
                    source="cloudflare_radar",
                    provider="market-wide",
                    confidence=0.7,
                ))
        except Exception:
            pass

        # Fetch traffic anomalies
        try:
            resp = await self.client.get(
                f"{self.BASE_URL}/traffic_anomalies",
                headers=headers,
                params={
                    "dateStart": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "dateEnd": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "location": "US",
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            await self.store_raw(data, "traffic_anomalies_US")

            for anomaly in data.get("result", {}).get("traffic_anomalies", []):
                ts_str = anomaly.get("startDate", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
                points.append(DataPoint(
                    timestamp_utc=ts,
                    metric_id="net.traffic_anomalies",
                    value=1.0,
                    unit="event",
                    region="US",
                    source="cloudflare_radar",
                    provider="market-wide",
                    confidence=0.6,
                    metadata_json=anomaly,
                ))
        except Exception:
            pass

        return points

    def _generate_demo_data(self) -> List[DataPoint]:
        """Generate demo traffic data."""
        import random
        points: List[DataPoint] = []
        now = datetime.now(timezone.utc)
        for h in range(168):
            ts = now - timedelta(hours=h)
            hour = ts.hour
            base = 1_000_000_000
            daily_factor = 1.0 + 0.3 * max(0, (hour - 8) * (22 - hour)) / 49
            trend = 1.0 + 0.002 * (168 - h)
            noise = random.gauss(0, 0.05)
            val = base * daily_factor * trend * (1 + noise)
            points.append(DataPoint(
                timestamp_utc=ts,
                metric_id="net.http_requests",
                value=round(val),
                unit="requests",
                region="US",
                source="demo",
                provider="market-wide",
                confidence=0.5,
            ))
        return points
