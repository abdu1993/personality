"""Base connector class. All connectors follow this interface."""
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import httpx
import json
import os
import hashlib


class DataPoint:
    """Normalized datapoint produced by connectors."""
    def __init__(
        self,
        timestamp_utc: datetime,
        metric_id: str,
        value: float,
        unit: str,
        region: str = "global",
        source: str = "",
        provider: str = "market-wide",
        confidence: float = 0.5,
        metadata_json: Optional[Dict] = None,
    ):
        self.timestamp_utc = timestamp_utc
        self.metric_id = metric_id
        self.value = value
        self.unit = unit
        self.region = region
        self.source = source
        self.provider = provider
        self.confidence = confidence
        self.metadata_json = metadata_json

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "metric_id": self.metric_id,
            "value": self.value,
            "unit": self.unit,
            "region": self.region,
            "source": self.source,
            "provider": self.provider,
            "confidence": self.confidence,
            "metadata_json": self.metadata_json,
        }


class BaseConnector(ABC):
    """All connectors inherit from this base."""

    name: str = "base"
    schema_version: str = "1.0"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self._raw_store_dir = "/tmp/connector_raw"
        os.makedirs(self._raw_store_dir, exist_ok=True)

    @abstractmethod
    async def fetch(self) -> List[DataPoint]:
        """Fetch and normalize data from the source."""
        ...

    async def store_raw(self, data: Any, label: str) -> str:
        """Store raw response JSON for auditability."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        content = json.dumps(data, default=str)
        h = hashlib.md5(content.encode()).hexdigest()[:8]
        path = os.path.join(self._raw_store_dir, f"{self.name}_{label}_{ts}_{h}.json.gz")
        import gzip
        with gzip.open(path, "wt") as f:
            f.write(content)
        return path

    async def close(self):
        await self.client.aclose()
