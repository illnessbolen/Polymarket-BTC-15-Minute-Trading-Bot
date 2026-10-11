"""
Order execution.

PaperTrader fills against the real order book that was just fetched (walking
the asks, charging Polymarket's taker fee), so paper results reflect the
prices the bot could actually have got. LiveTrader sends a fill-or-kill market
buy through py-clob-client.

Errors are split in two on purpose:
  OrderRejected     — the order definitely did not execute; safe to retry.
  OrderStateUnknown — the request may or may not have executed; never retry,
                      a human has to look at the account.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from polybot.config import Settings
from polybot.polymarket import OrderBook


class OrderRejected(Exception):
    pass


class OrderStateUnknown(Exception):
    pass


@dataclass(frozen=True)
class Fill:
    shares: float
    avg_price: float
    cost_usd: float
    fee_usd: float
    order_id: Optional[str] = None


def taker_fee(shares: float, price: float, rate: float) -> float:
    """Polymarket taker fee in USDC: shares × rate × p × (1 − p)."""
    return shares * rate * price * (1.0 - price)


def floor_to_tick(price: float, tick: float) -> float:
    ticks = math.floor(price / tick + 1e-9)
    return round(ticks * tick, 6)


class PaperTrader:
    mode = "paper"

    def __init__(self, fee_rate: float = 0.07):
        self.fee_rate = fee_rate

    async def buy(self, token_id: str, usd: float, book: OrderBook, max_price: float, tick_size: float) -> Fill:
        sim = book.simulate_buy(usd, max_price)
        if sim is None:
            raise OrderRejected(f"в стакане не хватает предложений на ${usd:.2f} по цене ≤ {max_price:.2f}")
        fee = sum(taker_fee(l.size, l.price, self.fee_rate) for l in sim.levels)
        return Fill(shares=sim.shares, avg_price=sim.avg_price, cost_usd=sim.cost_usd, fee_usd=fee)


class LiveTrader:
    mode = "live"

    def __init__(self, settings: Settings, fee_rate: float = 0.07):
        # Imported lazily so paper mode works without py-clob-client's web3 deps.
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds

        self.fee_rate = fee_rate
        self._client = ClobClient(
            settings.clob_url,
            chain_id=137,
            key=settings.private_key,
            signature_type=settings.signature_type,
            funder=settings.funder,
        )
        if settings.api_key and settings.api_secret and settings.api_passphrase:
            creds = ApiCreds(settings.api_key, settings.api_secret, settings.api_passphrase)
        else:
            logger.info("No Polymarket API credentials configured, deriving them from the private key")
            creds = self._client.create_or_derive_api_creds()
        self._client.set_api_creds(creds)

    async def balance_usd(self) -> Optional[float]:
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

        try:
            resp = await asyncio.to_thread(
                self._client.get_balance_allowance,
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
            )
            return int(resp["balance"]) / 1e6  # USDC has 6 decimals
        except Exception as e:
            logger.warning(f"Could not read USDC balance: {e}")
            return None

    async def buy(self, token_id: str, usd: float, book: OrderBook, max_price: float, tick_size: float) -> Fill:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.exceptions import PolyApiException
        from py_clob_client.order_builder.constants import BUY

        limit = floor_to_tick(min(max_price, 1.0 - tick_size), tick_size)
        args = MarketOrderArgs(token_id=token_id, amount=usd, side=BUY, price=limit, order_type=OrderType.FOK)
        try:
            # Signing is local; failures here mean nothing was sent.
            signed = await asyncio.to_thread(self._client.create_market_order, args)
        except Exception as e:
            raise OrderRejected(f"не удалось подписать ордер: {e}") from e

        try:
            resp = await asyncio.to_thread(self._client.post_order, signed, OrderType.FOK)
        except PolyApiException as e:
            if e.status_code is not None and 400 <= e.status_code < 500:
                raise OrderRejected(f"Polymarket отклонил ордер: {e.error_msg}") from e
            raise OrderStateUnknown(f"ошибка при отправке ордера, статус неизвестен: {e}") from e
        except Exception as e:
            raise OrderStateUnknown(f"ошибка при отправке ордера, статус неизвестен: {e}") from e

        if not isinstance(resp, dict) or not resp.get("success", False):
            msg = resp.get("errorMsg") if isinstance(resp, dict) else resp
            raise OrderRejected(f"Polymarket отклонил ордер: {msg}")

        order_id = resp.get("orderID") or resp.get("orderId")
        # For a BUY, makingAmount is USDC paid and takingAmount is shares received.
        try:
            cost = float(resp.get("makingAmount") or 0)
            shares = float(resp.get("takingAmount") or 0)
        except (TypeError, ValueError):
            cost = shares = 0.0
        if cost <= 0 or shares <= 0:
            sim = book.simulate_buy(usd, limit)
            cost = usd
            shares = sim.shares if sim else usd / limit
            logger.warning(f"Order {order_id}: fill amounts missing in response, using book estimate")
        avg_price = cost / shares
        return Fill(
            shares=shares,
            avg_price=avg_price,
            cost_usd=cost,
            # The protocol charges the fee at match time; this is an estimate kept
            # for the journal, so recorded live P&L errs on the conservative side.
            fee_usd=taker_fee(shares, avg_price, self.fee_rate),
            order_id=order_id,
        )
