"""EIA Hourly Electric Grid Monitor connector (EIA-930 data).

Fetches balancing authority demand data for regions that proxy data-center
power consumption (PJM / Dominion, ERCOT).
"""
from datetime import datetime, timezone, timedelta
from typing import List
from app.connectors.base import BaseConnector, DataPoint
from app.config import settings

# Balancing authorities of interest
BA_REGIONS = {
    "PJM": "PJM",       # Data Center Alley proxy
    "ERCO": "ERCOT",    # Texas data center corridor
}


class EIAConnector(BaseConnector):
    name = "eia_grid"
    schema_version = "1.0"

    BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

    async def fetch(self) -> List[DataPoint]:
        if not settings.eia_api_key:
            return self._generate_demo_data()

        points: List[DataPoint] = []
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=48)

        for ba_code, region_label in BA_REGIONS.items():
            params = {
                "api_key": settings.eia_api_key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": ba_code,
                "facets[type][]": "D",  # Demand
                "start": start.strftime("%Y-%m-%dT%H"),
                "end": end.strftime("%Y-%m-%dT%H"),
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "length": 200,
            }

            try:
                resp = await self.client.get(self.BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
                await self.store_raw(data, f"demand_{ba_code}")

                for row in data.get("response", {}).get("data", []):
                    ts_str = row.get("period", "")
                    try:
                        ts = datetime.strptime(ts_str, "%Y-%m-%dT%H").replace(
                            tzinfo=timezone.utc
                        )
                    except (ValueError, TypeError):
                        continue

                    val = row.get("value")
                    if val is None:
                        continue

                    points.append(DataPoint(
                        timestamp_utc=ts,
                        metric_id="grid.demand_mw",
                        value=float(val),
                        unit="MW",
                        region=region_label,
                        source="EIA-930",
                        provider="market-wide",
                        confidence=0.9,
                    ))
            except Exception:
                pass

        return points

    def _generate_demo_data(self) -> List[DataPoint]:
        """Generate realistic demo data when no API key is configured."""
        import random
        points: List[DataPoint] = []
        now = datetime.now(timezone.utc)
        for region, label in [("PJM", "PJM"), ("ERCO", "ERCOT")]:
            base_load = 85000 if label == "PJM" else 45000
            for h in range(168):  # 7 days
                ts = now - timedelta(hours=h)
                hour = ts.hour
                # Simulate daily cycle with growing base load
                daily_factor = 1.0 + 0.15 * max(0, (hour - 6) * (18 - hour)) / 36
                trend = 1.0 + 0.001 * (168 - h)
                noise = random.gauss(0, 0.02)
                val = base_load * daily_factor * trend * (1 + noise)
                points.append(DataPoint(
                    timestamp_utc=ts,
                    metric_id="grid.demand_mw",
                    value=round(val, 1),
                    unit="MW",
                    region=label,
                    source="demo",
                    provider="market-wide",
                    confidence=0.5,
                ))
        return points
