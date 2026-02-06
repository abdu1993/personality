"""GPU spot pricing connector.

Fetches AWS EC2 spot price history for GPU instance types.
Falls back to demo data if AWS credentials are not configured.
"""
from datetime import datetime, timezone, timedelta
from typing import List
from app.connectors.base import BaseConnector, DataPoint
from app.config import settings

# GPU instance types to track
GPU_INSTANCES = [
    {"type": "p5.48xlarge", "gpu": "H100", "gpu_count": 8},
    {"type": "p4d.24xlarge", "gpu": "A100", "gpu_count": 8},
    {"type": "g5.48xlarge", "gpu": "A10G", "gpu_count": 8},
    {"type": "g6.48xlarge", "gpu": "L4", "gpu_count": 8},
]

AWS_REGIONS = ["us-east-1", "us-west-2"]


class GPUSpotConnector(BaseConnector):
    name = "gpu_spot"
    schema_version = "1.0"

    async def fetch(self) -> List[DataPoint]:
        if not settings.aws_access_key_id:
            return self._generate_demo_data()

        points: List[DataPoint] = []

        try:
            import boto3
            for aws_region in AWS_REGIONS:
                client = boto3.client(
                    "ec2",
                    region_name=aws_region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                )

                for inst in GPU_INSTANCES:
                    try:
                        resp = client.describe_spot_price_history(
                            InstanceTypes=[inst["type"]],
                            ProductDescriptions=["Linux/UNIX"],
                            StartTime=datetime.now(timezone.utc) - timedelta(hours=24),
                            MaxResults=100,
                        )

                        raw = resp.get("SpotPriceHistory", [])
                        await self.store_raw(raw, f"spot_{inst['type']}_{aws_region}")

                        for record in raw:
                            ts = record["Timestamp"]
                            if isinstance(ts, str):
                                ts = datetime.fromisoformat(ts)
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)

                            price = float(record["SpotPrice"])
                            points.append(DataPoint(
                                timestamp_utc=ts,
                                metric_id="gpu_spot.price_usd_per_hour",
                                value=price,
                                unit="USD/hour",
                                region=aws_region,
                                source="aws_spot",
                                provider="Amazon/AWS",
                                confidence=0.95,
                                metadata_json={
                                    "instance_type": inst["type"],
                                    "gpu": inst["gpu"],
                                    "gpu_count": inst["gpu_count"],
                                    "az": record.get("AvailabilityZone", ""),
                                },
                            ))
                    except Exception:
                        continue
        except ImportError:
            return self._generate_demo_data()

        return points

    def _generate_demo_data(self) -> List[DataPoint]:
        """Generate demo GPU spot pricing data."""
        import random
        points: List[DataPoint] = []
        now = datetime.now(timezone.utc)

        prices = {
            "H100": {"base": 12.50, "vol": 0.15},
            "A100": {"base": 4.50, "vol": 0.10},
            "A10G": {"base": 1.80, "vol": 0.08},
            "L4": {"base": 1.20, "vol": 0.08},
        }

        for inst in GPU_INSTANCES:
            gpu = inst["gpu"]
            p = prices[gpu]
            for h in range(168):
                ts = now - timedelta(hours=h)
                trend = 1.0 + 0.0005 * (168 - h)  # slight uptrend (tightening)
                noise = random.gauss(0, p["vol"])
                val = max(0.5, p["base"] * trend * (1 + noise))
                points.append(DataPoint(
                    timestamp_utc=ts,
                    metric_id="gpu_spot.price_usd_per_hour",
                    value=round(val, 4),
                    unit="USD/hour",
                    region="us-east-1",
                    source="demo",
                    provider="Amazon/AWS",
                    confidence=0.5,
                    metadata_json={
                        "instance_type": inst["type"],
                        "gpu": gpu,
                        "gpu_count": inst["gpu_count"],
                    },
                ))
        return points
