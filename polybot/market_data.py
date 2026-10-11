"""
External context for the signal processors: BTC spot price (Coinbase) and the
Crypto Fear & Greed index (alternative.me). Both are optional — when a source
is unreachable the corresponding processor is simply skipped.
"""
from __future__ import annotations

import time
from typing import Optional

import httpx
from loguru import logger

COINBASE_TICKER = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
FEAR_GREED = "https://api.alternative.me/fng/"


class MarketData:
    def __init__(self, http: Optional[httpx.AsyncClient] = None, fear_greed_ttl: float = 1800.0):
        self._http = http or httpx.AsyncClient(timeout=5.0, headers={"User-Agent": "polybot/1.0"})
        self._fg_ttl = fear_greed_ttl
        self._fg_cache: Optional[tuple[float, dict]] = None

    async def aclose(self) -> None:
        await self._http.aclose()

    async def spot_price(self) -> Optional[float]:
        try:
            resp = await self._http.get(COINBASE_TICKER)
            resp.raise_for_status()
            return float(resp.json()["price"])
        except Exception as e:
            logger.warning(f"Coinbase spot price unavailable: {e}")
            return None

    async def fear_greed(self) -> Optional[dict]:
        """{'value': 0-100, 'classification': str}; the index updates daily, so it's cached."""
        if self._fg_cache and time.monotonic() - self._fg_cache[0] < self._fg_ttl:
            return self._fg_cache[1]
        try:
            resp = await self._http.get(FEAR_GREED)
            resp.raise_for_status()
            current = resp.json()["data"][0]
            value = {"value": float(current["value"]), "classification": current.get("value_classification", "")}
        except Exception as e:
            logger.warning(f"Fear & Greed index unavailable: {e}")
            return None
        self._fg_cache = (time.monotonic(), value)
        return value
