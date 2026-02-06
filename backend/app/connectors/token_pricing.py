"""Token pricing connector.

Tracks API pricing for major AI model providers. Since pricing pages
require scraping, this connector uses known pricing data and can be
supplemented with manual CSV uploads or automated scraping.
"""
from datetime import datetime, timezone, timedelta
from typing import List
from app.connectors.base import BaseConnector, DataPoint

# Current known pricing (USD per 1M tokens) — updated periodically
KNOWN_PRICES = [
    # OpenAI
    {"provider": "OpenAI", "model": "GPT-4o", "input": 2.50, "output": 10.00},
    {"provider": "OpenAI", "model": "GPT-4o-mini", "input": 0.15, "output": 0.60},
    {"provider": "OpenAI", "model": "o1", "input": 15.00, "output": 60.00},
    # Anthropic
    {"provider": "Anthropic", "model": "Claude-3.5-Sonnet", "input": 3.00, "output": 15.00},
    {"provider": "Anthropic", "model": "Claude-3-Haiku", "input": 0.25, "output": 1.25},
    # Google
    {"provider": "Google", "model": "Gemini-1.5-Pro", "input": 1.25, "output": 5.00},
    {"provider": "Google", "model": "Gemini-1.5-Flash", "input": 0.075, "output": 0.30},
    {"provider": "Google", "model": "Gemini-2.0-Flash", "input": 0.10, "output": 0.40},
    # Amazon
    {"provider": "Amazon", "model": "Nova-Pro", "input": 0.80, "output": 3.20},
    {"provider": "Amazon", "model": "Nova-Lite", "input": 0.06, "output": 0.24},
]


class TokenPricingConnector(BaseConnector):
    name = "token_pricing"
    schema_version = "1.0"

    async def fetch(self) -> List[DataPoint]:
        """Snapshot current known pricing into time series."""
        now = datetime.now(timezone.utc)
        points: List[DataPoint] = []

        for entry in KNOWN_PRICES:
            # Input token price
            points.append(DataPoint(
                timestamp_utc=now,
                metric_id="model_price.input_usd_per_1m_tokens",
                value=entry["input"],
                unit="USD/1M_tokens",
                region="global",
                source="manual_pricing_table",
                provider=entry["provider"],
                confidence=0.85,
                metadata_json={"model": entry["model"]},
            ))
            # Output token price
            points.append(DataPoint(
                timestamp_utc=now,
                metric_id="model_price.output_usd_per_1m_tokens",
                value=entry["output"],
                unit="USD/1M_tokens",
                region="global",
                source="manual_pricing_table",
                provider=entry["provider"],
                confidence=0.85,
                metadata_json={"model": entry["model"]},
            ))

        await self.store_raw(
            [e for e in KNOWN_PRICES], "pricing_snapshot"
        )
        return points
