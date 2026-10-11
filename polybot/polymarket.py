"""
Read-only Polymarket API client: finds the BTC 15-minute "Up or Down" market
for a given window (Gamma API) and reads its order book (CLOB API).

Each market is an event with slug ``btc-updown-15m-<start_unix_ts>``; start
timestamps are aligned to 15-minute UTC boundaries. Gamma's /events endpoint
returns both open and closed markets, so the same call is used to discover
a market and to learn how it resolved.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import httpx

INTERVAL_SECONDS = 900
SLUG_PREFIX = "btc-updown-15m-"


def window_start(ts: float) -> int:
    """Start (unix seconds) of the 15-minute window containing ``ts``."""
    return int(ts // INTERVAL_SECONDS) * INTERVAL_SECONDS


def slug_for(start_ts: int) -> str:
    return f"{SLUG_PREFIX}{start_ts}"


def _json_list(value) -> list:
    """Gamma returns some list fields as JSON-encoded strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


@dataclass(frozen=True)
class Market:
    slug: str
    question: str
    condition_id: str
    start_ts: int
    end_ts: int
    up_token_id: str
    down_token_id: str
    tick_size: float
    min_order_size: float
    accepting_orders: bool
    closed: bool
    winner: Optional[str]  # "UP" / "DOWN" once resolved, else None

    def token_for(self, side: str) -> str:
        return self.up_token_id if side == "UP" else self.down_token_id

    @classmethod
    def from_gamma(cls, market: dict, start_ts: int) -> "Market":
        outcomes = [str(o).strip().lower() for o in _json_list(market.get("outcomes"))]
        tokens = [str(t) for t in _json_list(market.get("clobTokenIds"))]
        if len(tokens) != 2 or sorted(outcomes) != ["down", "up"]:
            raise ValueError(f"unexpected market shape: outcomes={outcomes} tokens={len(tokens)}")
        by_outcome = dict(zip(outcomes, tokens))

        closed = bool(market.get("closed"))
        winner = None
        if closed:
            prices = _json_list(market.get("outcomePrices"))
            for outcome, price in zip(outcomes, prices):
                try:
                    if float(price) >= 0.99:
                        winner = outcome.upper()
                except (TypeError, ValueError):
                    continue

        return cls(
            slug=market.get("slug") or slug_for(start_ts),
            question=market.get("question") or "",
            condition_id=market.get("conditionId") or "",
            start_ts=start_ts,
            end_ts=start_ts + INTERVAL_SECONDS,
            up_token_id=by_outcome["up"],
            down_token_id=by_outcome["down"],
            tick_size=float(market.get("orderPriceMinTickSize") or 0.01),
            min_order_size=float(market.get("orderMinSize") or 5),
            accepting_orders=bool(market.get("acceptingOrders")),
            closed=closed,
            winner=winner,
        )


@dataclass(frozen=True)
class Level:
    price: float
    size: float


@dataclass(frozen=True)
class SimulatedFill:
    shares: float
    avg_price: float
    cost_usd: float
    levels: tuple[Level, ...]  # (price, shares taken) per level


@dataclass(frozen=True)
class OrderBook:
    token_id: str
    bids: tuple[Level, ...]  # best (highest) first
    asks: tuple[Level, ...]  # best (lowest) first

    @classmethod
    def from_clob(cls, token_id: str, data: dict) -> "OrderBook":
        def levels(raw) -> list[Level]:
            out = []
            for lvl in raw or []:
                try:
                    price, size = float(lvl["price"]), float(lvl["size"])
                except (KeyError, TypeError, ValueError):
                    continue
                if price > 0 and size > 0:
                    out.append(Level(price, size))
            return out

        # The CLOB lists bids ascending and asks descending (best price last);
        # sort explicitly so nothing depends on that ordering.
        bids = sorted(levels(data.get("bids")), key=lambda l: l.price, reverse=True)
        asks = sorted(levels(data.get("asks")), key=lambda l: l.price)
        return cls(token_id=token_id, bids=tuple(bids), asks=tuple(asks))

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    def simulate_buy(self, usd: float, max_price: float) -> Optional[SimulatedFill]:
        """
        Walk the asks to spend ``usd`` without paying more than ``max_price``.
        Returns None when the book can't absorb the whole amount (FOK semantics).
        """
        remaining = usd
        shares = 0.0
        taken: list[Level] = []
        for lvl in self.asks:
            if lvl.price > max_price + 1e-9 or remaining <= 1e-9:
                break
            level_cost = lvl.price * lvl.size
            if remaining <= level_cost:
                qty = remaining / lvl.price
                remaining = 0.0
            else:
                qty = lvl.size
                remaining -= level_cost
            shares += qty
            taken.append(Level(lvl.price, qty))
        if remaining > 1e-9 or shares <= 0:
            return None
        return SimulatedFill(shares=shares, avg_price=usd / shares, cost_usd=usd, levels=tuple(taken))


class PolymarketClient:
    def __init__(
        self,
        gamma_url: str = "https://gamma-api.polymarket.com",
        clob_url: str = "https://clob.polymarket.com",
        http: Optional[httpx.AsyncClient] = None,
    ):
        self.gamma_url = gamma_url.rstrip("/")
        self.clob_url = clob_url.rstrip("/")
        self._http = http or httpx.AsyncClient(
            timeout=10.0, headers={"User-Agent": "polybot/1.0"}
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get_market(self, start_ts: int) -> Optional[Market]:
        """Market for the window starting at ``start_ts``, or None if it doesn't exist."""
        resp = await self._http.get(f"{self.gamma_url}/events", params={"slug": slug_for(start_ts)})
        resp.raise_for_status()
        events = resp.json()
        if not events:
            return None
        markets = events[0].get("markets") or []
        if not markets:
            return None
        return Market.from_gamma(markets[0], start_ts)

    async def get_book(self, token_id: str) -> OrderBook:
        resp = await self._http.get(f"{self.clob_url}/book", params={"token_id": token_id})
        resp.raise_for_status()
        return OrderBook.from_clob(token_id, resp.json())
