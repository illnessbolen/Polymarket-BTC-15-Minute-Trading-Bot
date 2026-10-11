"""
Main loop.

Every POLL_INTERVAL seconds the bot:
  1. makes sure it is looking at the market for the current 15-minute window,
  2. reads the Up order book and records the mid price (feeds the processors),
  3. inside the trade window (minute 13 by default) decides once per market
     and, if the decision is to trade, buys Up or Down,
  4. every RESOLVE_INTERVAL seconds settles open trades whose market resolved.

Telegram commands are served concurrently from the same event loop.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from html import escape
from typing import Callable, Optional

from loguru import logger

from polybot import messages
from polybot.config import Settings
from polybot.market_data import MarketData
from polybot.polymarket import INTERVAL_SECONDS, Market, OrderBook, PolymarketClient, window_start
from polybot.storage import Storage
from polybot.strategy import Evaluation, SignalEngine, Snapshot, decide
from polybot.telegram import Telegram
from polybot.trader import OrderRejected, OrderStateUnknown

ERROR_ALERT_AFTER = 12  # consecutive failed polls before alerting in Telegram


def utc_midnight_iso(ts: float) -> str:
    day = datetime.fromtimestamp(ts, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return day.isoformat(timespec="seconds")


class Bot:
    def __init__(
        self,
        settings: Settings,
        polymarket: PolymarketClient,
        trader,
        storage: Storage,
        telegram: Telegram,
        market_data: MarketData,
        signal_engine: SignalEngine,
        clock: Callable[[], float] = time.time,
    ):
        self.settings = settings
        self.pm = polymarket
        self.trader = trader
        self.storage = storage
        self.telegram = telegram
        self.market_data = market_data
        self.signal_engine = signal_engine
        self.clock = clock

        self.market: Optional[Market] = None
        self.up_book: Optional[OrderBook] = None
        self.price_history: deque[float] = deque(maxlen=100)
        self.ticks: deque[dict] = deque(maxlen=500)
        self.last_decision: Optional[str] = None

        self._market_done = False
        self._attempts = 0
        self._missing_market_logged: Optional[int] = None
        self._loss_limit_day: Optional[str] = None
        self._last_resolve = 0.0
        self._errors_in_row = 0
        self._eval_lock = asyncio.Lock()

    # --- state ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self.trader.mode

    @property
    def paused(self) -> bool:
        return self.storage.get_state("paused") == "1"

    def set_paused(self, paused: bool) -> None:
        self.storage.set_state("paused", "1" if paused else "0")

    # --- market tracking ----------------------------------------------------------

    async def _switch_market(self, start_ts: int) -> None:
        market = await self.pm.get_market(start_ts)
        if market is None:
            if self._missing_market_logged != start_ts:
                logger.warning(f"No BTC 15m market found for window {start_ts}")
                self._missing_market_logged = start_ts
            self.market = None
            return
        self.market = market
        self.up_book = None
        self.price_history.clear()
        self.ticks.clear()
        self._market_done = False
        self._attempts = 0
        self.last_decision = None
        logger.info(f"Tracking market {market.slug}: {market.question}")

    async def step(self) -> None:
        now = self.clock()
        start = window_start(now)
        if self.market is None or self.market.start_ts != start:
            await self._switch_market(start)
        if self.market is None:
            return

        self.up_book = await self.pm.get_book(self.market.up_token_id)
        mid = self.up_book.mid
        if mid is not None:
            self.price_history.append(mid)
            self.ticks.append({"ts": datetime.fromtimestamp(now, timezone.utc), "price": mid})

        elapsed = now - self.market.start_ts
        s = self.settings
        if not self._market_done and s.trade_window_start <= elapsed < s.trade_window_end:
            await self._try_trade(now)

    # --- trading ------------------------------------------------------------------

    async def evaluate(self, up_price: float) -> Evaluation:
        async with self._eval_lock:
            spot, fear_greed = await asyncio.gather(
                self.market_data.spot_price(), self.market_data.fear_greed()
            )
            snap = Snapshot(
                up_price=up_price,
                price_history=list(self.price_history),
                tick_buffer=list(self.ticks),
                up_token_id=self.market.up_token_id,
                spot_price=spot,
                fear_greed=fear_greed,
            )
            return await asyncio.to_thread(self.signal_engine.evaluate, snap)

    async def _finish(self, reason: str, notify: bool = False) -> None:
        """Stop considering the current market."""
        self._market_done = True
        self.last_decision = reason
        logger.info(f"{self.market.slug}: {reason}")
        if notify:
            await self.telegram.send(messages.skipped(self.market.question, reason))

    def daily_pnl(self, now: float) -> float:
        return self.storage.stats(self.mode, since=utc_midnight_iso(now))["pnl"]

    async def _try_trade(self, now: float) -> None:
        s = self.settings
        m = self.market

        if self.paused:
            # Don't finish the market: /resume inside the window can still trade it.
            self.last_decision = "бот на паузе"
            return
        if not m.accepting_orders:
            return await self._finish("рынок не принимает ордера")
        if self.storage.has_trade(m.slug, self.mode):
            return await self._finish("по этому рынку уже есть сделка")
        if s.max_daily_loss_usd > 0:
            today = self.daily_pnl(now)
            if today <= -s.max_daily_loss_usd:
                day = utc_midnight_iso(now)
                notify = self._loss_limit_day != day
                self._loss_limit_day = day
                return await self._finish(
                    f"дневной лимит убытка достигнут ({messages.usd(today)} ≤ −${s.max_daily_loss_usd:.2f})",
                    notify=notify,
                )

        up_price = self.up_book.mid if self.up_book else None
        if up_price is None:
            self.last_decision = "нет двусторонней котировки Up, жду"
            return

        evaluation = await self.evaluate(up_price)
        decision = decide(up_price, evaluation, s)
        if decision.side is None:
            return await self._finish(f"пропуск: {decision.reason}", notify=s.notify_skips)

        side = decision.side
        token_id = m.token_for(side)
        book = self.up_book if side == "UP" else await self.pm.get_book(token_id)
        ask = book.best_ask
        if ask is None:
            return await self._retry_or_finish(f"в стакане {side} нет продавцов")
        if ask > s.max_entry_price:
            return await self._finish(
                f"пропуск: цена {side} {ask:.2f} выше MAX_ENTRY_PRICE {s.max_entry_price:.2f}",
                notify=s.notify_skips,
            )

        max_price = min(s.max_entry_price, ask + s.slippage)
        logger.info(f"{m.slug}: {decision.reason} -> buying {side} for ${s.trade_size_usd:.2f} (ask {ask:.3f})")
        try:
            fill = await self.trader.buy(token_id, s.trade_size_usd, book, max_price, m.tick_size)
        except OrderRejected as e:
            return await self._retry_or_finish(f"ордер отклонён: {e}")
        except OrderStateUnknown as e:
            self._market_done = True
            self.last_decision = f"статус ордера неизвестен: {e}"
            logger.error(f"{m.slug}: {e}")
            await self.telegram.send(
                f"🚨 <b>Статус ордера неизвестен</b> ({escape(m.slug)})\n{escape(str(e))}\n"
                f"Проверьте позиции на polymarket.com. Повторных попыток по этому рынку не будет."
            )
            return

        # Mark the market done before anything else can fail: a filled order
        # must never be followed by a second one on the same market.
        self._market_done = True
        summary = evaluation.summary()
        summary.update({"up_price": round(up_price, 4), "reason": decision.reason})
        trade = self.storage.add_trade(
            mode=self.mode,
            market_slug=m.slug,
            market_end_ts=m.end_ts,
            side=side,
            token_id=token_id,
            price=fill.avg_price,
            size_usd=fill.cost_usd,
            shares=fill.shares,
            fee_usd=fill.fee_usd,
            order_id=fill.order_id,
            signals=summary,
        )
        self.last_decision = f"куплено {side} по {fill.avg_price:.3f}"
        logger.success(
            f"[{self.mode.upper()}] Trade #{trade.id}: {side} {fill.shares:.3f} sh @ {fill.avg_price:.3f} "
            f"(${fill.cost_usd:.2f}, fee ${fill.fee_usd:.3f})"
        )
        await self.telegram.send(messages.trade_opened(trade, m.question, decision.reason))

    async def _retry_or_finish(self, reason: str) -> None:
        self._attempts += 1
        if self._attempts >= self.settings.max_attempts_per_market:
            await self._finish(f"{reason} (попыток: {self._attempts})", notify=True)
        else:
            self.last_decision = f"{reason}, повтор на следующем тике"
            logger.warning(f"{self.market.slug}: {self.last_decision}")

    # --- settlement -----------------------------------------------------------------

    async def resolve(self) -> None:
        now = self.clock()
        for trade in self.storage.open_trades():
            if now < trade.market_end_ts:
                continue
            try:
                market = await self.pm.get_market(trade.market_end_ts - INTERVAL_SECONDS)
            except Exception as e:
                logger.warning(f"Could not check resolution of {trade.market_slug}: {e}")
                continue
            if market is None or market.winner is None:
                continue
            settled = self.storage.settle(trade.id, won=trade.side == market.winner)
            totals = self.storage.stats(settled.mode)
            logger.info(
                f"[{settled.mode.upper()}] Trade #{settled.id} {settled.status}: "
                f"{settled.side} vs {market.winner}, P&L {settled.pnl_usd:+.2f}"
            )
            await self.telegram.send(messages.trade_resolved(settled, market.winner, totals))

    # --- main loop --------------------------------------------------------------------

    async def run(self, stop: asyncio.Event) -> None:
        s = self.settings
        settings_line = (
            f"Вход на {s.trade_window_start // 60}-й минуте, если Up > {s.trend_up_threshold:.2f} "
            f"или < {s.trend_down_threshold:.2f}; фильтр сигналов: {s.signal_filter}; "
            f"макс. цена {s.max_entry_price:.2f}; дневной лимит убытка ${s.max_daily_loss_usd:.2f}"
        )
        logger.info(f"Polybot started in {self.mode.upper()} mode. {settings_line}")
        await self.telegram.send(messages.startup(self.mode, s.trade_size_usd, settings_line, self.paused))

        telegram_task = asyncio.create_task(self.telegram.run(self.handle_command, stop))
        try:
            while not stop.is_set():
                started = self.clock()
                try:
                    await self.step()
                    if self._errors_in_row >= ERROR_ALERT_AFTER:
                        await self.telegram.send("✅ Связь с Polymarket восстановлена.")
                    self._errors_in_row = 0
                except Exception as e:
                    self._errors_in_row += 1
                    logger.warning(f"Poll failed ({self._errors_in_row} in a row): {e!r}")
                    if self._errors_in_row == ERROR_ALERT_AFTER:
                        await self.telegram.send(
                            f"⚠️ Не удаётся получить данные Polymarket уже {self._errors_in_row} раз подряд: "
                            f"{escape(repr(e))}"
                        )

                if self.clock() - self._last_resolve >= s.resolve_interval:
                    self._last_resolve = self.clock()
                    try:
                        await self.resolve()
                    except Exception as e:
                        logger.warning(f"Resolution check failed: {e!r}")

                delay = max(0.5, s.poll_interval - (self.clock() - started))
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
        finally:
            telegram_task.cancel()
            await asyncio.gather(telegram_task, return_exceptions=True)
            logger.info("Polybot stopped")
            await self.telegram.send("🛑 Polybot остановлен.")

    # --- Telegram commands --------------------------------------------------------------

    async def handle_command(self, command: str, arg: str, chat_id: int) -> Optional[str]:
        if command in ("start", "help"):
            return messages.HELP
        if command == "status":
            return self.status_text()
        if command == "stats":
            now = self.clock()
            modes = ("paper", "live")
            return messages.stats(
                {m: self.storage.stats(m) for m in modes},
                {m: self.storage.stats(m, since=utc_midnight_iso(now)) for m in modes},
            )
        if command == "trades":
            return messages.trades(self.storage.recent(10))
        if command == "pause":
            self.set_paused(True)
            return "⏸ Пауза: новые сделки не открываются. Открытые сделки будут рассчитаны как обычно."
        if command == "resume":
            self.set_paused(False)
            return "▶️ Торговля возобновлена."
        if command == "signals":
            return await self.signals_text()
        return "Неизвестная команда. /help — список команд."

    def status_text(self) -> str:
        s = self.settings
        now = self.clock()
        lines = [
            f"🤖 <b>Polybot</b> — {messages.MODE_LABEL[self.mode]}",
            "Состояние: " + ("⏸ пауза" if self.paused else "▶️ торгует"),
        ]
        m = self.market
        if m is None:
            lines.append("Рынок: не найден, ищу…")
        else:
            elapsed = now - m.start_ts
            lines.append(f"Рынок: {escape(m.question)}")
            book = self.up_book
            if book and book.mid is not None:
                lines.append(f"Up: bid {book.best_bid:.2f} / ask {book.best_ask:.2f} (mid {book.mid:.3f})")
            lines.append(f"Прошло {messages.mmss(elapsed)} из 15:00")
            if elapsed < s.trade_window_start:
                lines.append(f"Окно входа через {messages.mmss(s.trade_window_start - elapsed)}")
            elif elapsed < s.trade_window_end and not self._market_done:
                lines.append("Окно входа открыто")
            if self.last_decision:
                lines.append(f"Решение: {escape(self.last_decision)}")
        today = self.storage.stats(self.mode, since=utc_midnight_iso(now))
        lines.append(
            f"Сегодня: {messages.usd(today['pnl'])}, сделок {today['total']}, ждут итога {today['open']}"
        )
        return "\n".join(lines)

    async def signals_text(self) -> str:
        if self.market is None or self.up_book is None or self.up_book.mid is None:
            return "Нет данных по текущему рынку, попробуйте через несколько секунд."
        up_price = self.up_book.mid
        evaluation = await self.evaluate(up_price)
        decision = decide(up_price, evaluation, self.settings)
        verdict = f"купить {decision.side}" if decision.side else "пропустить"
        return (
            f"📡 <b>Сигналы</b> — {escape(self.market.question)}\n"
            f"Up mid: {up_price:.3f}, тиков в буфере: {len(self.ticks)}\n"
            f"{messages.signals_block(evaluation.summary())}\n"
            f"Если бы окно входа было сейчас: <b>{verdict}</b> ({escape(decision.reason)})"
        )
