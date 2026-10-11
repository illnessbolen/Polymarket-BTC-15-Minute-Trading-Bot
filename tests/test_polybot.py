"""
Offline tests for polybot. Run from the repository root:

    pip install -r requirements-bot.txt pytest
    python -m pytest tests -q
"""
import asyncio
import json
from datetime import datetime

import httpx
import pytest

from core.strategy_brain.fusion_engine.signal_fusion import FusedSignal
from core.strategy_brain.signal_processors.base_processor import SignalDirection
from core.strategy_brain.signal_processors.orderbook_processor import OrderBookImbalanceProcessor
from polybot.config import Settings
from polybot.engine import Bot
from polybot.polymarket import Market, OrderBook, slug_for, window_start
from polybot.storage import Storage
from polybot.strategy import Evaluation, SignalEngine, Snapshot, decide
from polybot.telegram import Telegram
from polybot.trader import OrderRejected, PaperTrader, floor_to_tick, taker_fee

START = 1_791_741_600  # a 15-minute boundary


# --- helpers -------------------------------------------------------------------------


def gamma_market(start=START, closed=False, prices=("0.5", "0.5"), outcomes=("Up", "Down")):
    return {
        "slug": slug_for(start),
        "question": "Bitcoin Up or Down - test",
        "conditionId": "0xabc",
        "outcomes": json.dumps(list(outcomes)),
        "clobTokenIds": json.dumps(["tok-" + outcomes[0].lower(), "tok-" + outcomes[1].lower()]),
        "outcomePrices": json.dumps(list(prices)),
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 5,
        "acceptingOrders": not closed,
        "closed": closed,
    }


def book(token, bids, asks):
    return OrderBook.from_clob(token, {
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
    })


def fused(direction):
    return FusedSignal(
        timestamp=datetime.now(), direction=direction, confidence=0.7, score=80.0,
        signals=[], weights={}, metadata={},
    )


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class FakePolymarket:
    def __init__(self, books, winner=None):
        self.books = books
        self.winner = winner

    async def get_market(self, start_ts):
        closed = self.winner is not None
        prices = ("1", "0") if self.winner == "UP" else ("0", "1") if self.winner == "DOWN" else ("0.5", "0.5")
        return Market.from_gamma(gamma_market(start_ts, closed=closed, prices=prices), start_ts)

    async def get_book(self, token_id):
        return self.books[token_id]


class FakeTelegram:
    def __init__(self):
        self.sent = []

    async def send(self, text, chat_id=None):
        self.sent.append(text)

    async def run(self, handler, stop):
        return None


class FakeMarketData:
    async def spot_price(self):
        return None

    async def fear_greed(self):
        return None


class FakeSignals:
    def __init__(self, evaluation=None):
        self.evaluation = evaluation or Evaluation(fused=fused(SignalDirection.BULLISH))
        self.calls = 0

    def evaluate(self, snap):
        self.calls += 1
        return self.evaluation


def make_bot(books, clock, settings=None, trader=None, signals=None, winner=None):
    settings = settings or Settings()
    tg = FakeTelegram()
    bot = Bot(
        settings=settings,
        polymarket=FakePolymarket(books, winner),
        trader=trader or PaperTrader(fee_rate=0.07),
        storage=Storage(":memory:"),
        telegram=tg,
        market_data=FakeMarketData(),
        signal_engine=signals or FakeSignals(),
        clock=clock,
    )
    return bot, tg


UP_STRONG = {
    "tok-up": book("tok-up", bids=[(0.70, 100)], asks=[(0.72, 100)]),
    "tok-down": book("tok-down", bids=[(0.28, 100)], asks=[(0.30, 100)]),
}


# --- polymarket parsing -----------------------------------------------------------------


def test_window_and_slug():
    assert window_start(START + 899.9) == START
    assert window_start(START + 900) == START + 900
    assert slug_for(START) == f"btc-updown-15m-{START}"


def test_market_parsing_open_and_resolved():
    m = Market.from_gamma(gamma_market(), START)
    assert (m.up_token_id, m.down_token_id) == ("tok-up", "tok-down")
    assert m.end_ts == START + 900 and m.winner is None and m.accepting_orders

    # Outcomes listed in the other order must still map correctly.
    swapped = Market.from_gamma(gamma_market(outcomes=("Down", "Up"), closed=True, prices=("1", "0")), START)
    assert swapped.up_token_id == "tok-up" and swapped.winner == "DOWN"

    # Live prices on an open market are not a resolution.
    assert Market.from_gamma(gamma_market(prices=("0.995", "0.005")), START).winner is None


def test_market_parsing_rejects_unexpected_shape():
    bad = gamma_market()
    bad["outcomes"] = json.dumps(["Yes", "No"])
    with pytest.raises(ValueError):
        Market.from_gamma(bad, START)


def test_order_book_sorted_best_first():
    # CLOB order: bids ascending, asks descending
    b = book("t", bids=[(0.01, 10), (0.40, 5), (0.39, 7)], asks=[(0.99, 10), (0.42, 3), (0.41, 2)])
    assert b.best_bid == 0.40 and b.best_ask == 0.41
    assert b.mid == pytest.approx(0.405)
    assert book("t", bids=[], asks=[(0.5, 1)]).mid is None


def test_simulate_buy_walks_levels_and_respects_limit():
    b = book("t", bids=[], asks=[(0.50, 2), (0.60, 10)])
    fill = b.simulate_buy(4.0, max_price=0.60)  # $1 at 0.50 (2 sh) + $3 at 0.60 (5 sh)
    assert fill.shares == pytest.approx(7.0)
    assert fill.avg_price == pytest.approx(4.0 / 7.0)
    assert b.simulate_buy(4.0, max_price=0.55) is None   # only $1 available under the limit
    assert b.simulate_buy(100.0, max_price=0.99) is None  # not enough depth


def test_fee_and_tick_helpers():
    # Polymarket docs: 100 shares at $0.50 on a crypto market -> $1.75 fee
    assert taker_fee(100, 0.50, 0.07) == pytest.approx(1.75)
    assert taker_fee(100, 0.30, 0.07) == pytest.approx(taker_fee(100, 0.70, 0.07))
    assert floor_to_tick(0.747, 0.01) == 0.74
    assert floor_to_tick(0.75, 0.01) == 0.75


# --- strategy ------------------------------------------------------------------------------


def test_decide_trend_filter():
    s = Settings(signal_filter="none")
    empty = Evaluation()
    assert decide(0.72, empty, s).side == "UP"
    assert decide(0.31, empty, s).side == "DOWN"
    assert decide(0.55, empty, s).side is None


def test_decide_signal_filters():
    bull = Evaluation(fused=fused(SignalDirection.BULLISH))
    assert decide(0.72, Evaluation(), Settings(signal_filter="any")).side is None
    assert decide(0.72, bull, Settings(signal_filter="any")).side == "UP"
    assert decide(0.31, bull, Settings(signal_filter="any")).side == "DOWN"  # any signal is enough
    assert decide(0.31, bull, Settings(signal_filter="agree")).side is None  # must agree with trend
    assert decide(0.72, bull, Settings(signal_filter="agree")).side == "UP"


def test_signal_engine_runs_repo_processors(monkeypatch):
    engine = SignalEngine()
    by_name = {p.name: p for p in engine.processors}
    # Keep the test offline: stub the two processors that call HTTP APIs.
    monkeypatch.setattr(by_name["OrderBookImbalance"], "fetch_order_book",
                        lambda token: {"bids": [{"price": "0.7", "size": "900"}],
                                       "asks": [{"price": "0.72", "size": "50"}]})
    monkeypatch.setattr(by_name["DeribitPCR"], "_fetch_pcr", lambda: None)

    prices = [0.60 + 0.005 * i for i in range(25)]
    evaluation = engine.evaluate(Snapshot(up_price=prices[-1], price_history=prices,
                                          tick_buffer=[], up_token_id="tok-up"))
    sources = {s.source for s in evaluation.signals}
    assert "OrderBookImbalance" in sources
    assert evaluation.fused is not None
    summary = evaluation.summary()
    json.dumps(summary)  # must be storable
    assert summary["fused"]["direction"] in ("BULLISH", "BEARISH")


def test_orderbook_processor_uses_levels_nearest_the_market():
    proc = OrderBookImbalanceProcessor(top_levels=2, min_book_volume=1)
    # Near the market: strong bid side. Far away (1c / 99c): huge ask wall that must be ignored.
    raw = {
        "bids": [{"price": "0.01", "size": "100"}, {"price": "0.60", "size": "500"}, {"price": "0.61", "size": "500"}],
        "asks": [{"price": "0.99", "size": "5000"}, {"price": "0.63", "size": "10"}, {"price": "0.62", "size": "10"}],
    }
    proc.fetch_order_book = lambda token: raw
    signal = proc.process(current_price=0.615, historical_prices=[], metadata={"yes_token_id": "tok"})
    assert signal is not None and signal.direction == SignalDirection.BULLISH


# --- storage -----------------------------------------------------------------------------------


def test_storage_settle_and_stats():
    st = Storage(":memory:")
    t = st.add_trade(mode="paper", market_slug="m1", market_end_ts=START + 900, side="UP",
                     token_id="tok-up", price=0.8, size_usd=1.0, shares=1.25, fee_usd=0.01)
    assert st.has_trade("m1", "paper") and not st.has_trade("m1", "live")
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):  # one trade per market and mode
        st.add_trade(mode="paper", market_slug="m1", market_end_ts=START + 900, side="DOWN",
                     token_id="tok-down", price=0.2, size_usd=1.0, shares=5)
    won = st.settle(t.id, won=True)
    assert won.status == "WON" and won.pnl_usd == pytest.approx(0.24)

    t2 = st.add_trade(mode="paper", market_slug="m2", market_end_ts=START + 1800, side="DOWN",
                      token_id="tok-down", price=0.7, size_usd=1.0, shares=1.4, fee_usd=0.02)
    st.settle(t2.id, won=False)
    stats = st.stats("paper")
    assert (stats["total"], stats["won"], stats["lost"], stats["open"]) == (2, 1, 1, 0)
    assert stats["pnl"] == pytest.approx(0.24 - 1.02)
    assert stats["win_rate"] == 0.5
    assert st.stats("live")["total"] == 0

    st.set_state("paused", "1")
    st.set_state("paused", "0")
    assert st.get_state("paused") == "0"


# --- engine ------------------------------------------------------------------------------------


def test_bot_trades_once_in_window_and_settles():
    clock = Clock(START + 100)
    bot, tg = make_bot(UP_STRONG, clock)

    asyncio.run(bot.step())  # minute 1: just tracking
    assert bot.storage.stats("paper")["total"] == 0
    assert bot.market.slug == slug_for(START)

    clock.t = START + 790  # minute 13
    asyncio.run(bot.step())
    trades = bot.storage.recent()
    assert len(trades) == 1
    t = trades[0]
    assert (t.side, t.mode, t.token_id) == ("UP", "paper", "tok-up")
    assert t.price == pytest.approx(0.72)
    assert t.fee_usd == pytest.approx(taker_fee(1 / 0.72, 0.72, 0.07))
    assert "Сделка открыта" in tg.sent[-1]

    clock.t = START + 800  # still in window: no second trade
    asyncio.run(bot.step())
    assert len(bot.storage.recent()) == 1

    # Market ends and resolves Up.
    clock.t = START + 960
    bot.pm.winner = "UP"
    asyncio.run(bot.resolve())
    settled = bot.storage.get_trade(t.id)
    assert settled.status == "WON"
    assert settled.pnl_usd == pytest.approx(1 / 0.72 - 1.0 - t.fee_usd)
    assert "Выигрыш" in tg.sent[-1]


def test_bot_buys_down_token_when_trend_is_down():
    books = {
        "tok-up": book("tok-up", bids=[(0.25, 100)], asks=[(0.27, 100)]),
        "tok-down": book("tok-down", bids=[(0.73, 100)], asks=[(0.75, 100)]),
    }
    bot, _ = make_bot(books, Clock(START + 790))
    asyncio.run(bot.step())
    t = bot.storage.recent()[0]
    assert (t.side, t.token_id) == ("DOWN", "tok-down") and t.price == pytest.approx(0.75)


def test_bot_skips_neutral_price_and_expensive_entry():
    neutral = {"tok-up": book("tok-up", bids=[(0.49, 100)], asks=[(0.51, 100)]), "tok-down": UP_STRONG["tok-down"]}
    bot, _ = make_bot(neutral, Clock(START + 790))
    asyncio.run(bot.step())
    assert bot.storage.stats("paper")["total"] == 0 and "нейтральной" in bot.last_decision

    pricey = {"tok-up": book("tok-up", bids=[(0.96, 100)], asks=[(0.97, 100)]), "tok-down": UP_STRONG["tok-down"]}
    bot, _ = make_bot(pricey, Clock(START + 790))
    asyncio.run(bot.step())
    assert bot.storage.stats("paper")["total"] == 0 and "MAX_ENTRY_PRICE" in bot.last_decision


def test_bot_respects_pause_and_daily_loss_limit():
    clock = Clock(START + 790)
    bot, _ = make_bot(UP_STRONG, clock)
    bot.set_paused(True)
    asyncio.run(bot.step())
    assert bot.storage.stats("paper")["total"] == 0
    bot.set_paused(False)  # resumed while the window is still open
    clock.t += 5
    asyncio.run(bot.step())
    assert bot.storage.stats("paper")["total"] == 1

    bot, tg = make_bot(UP_STRONG, Clock(START + 790), settings=Settings(max_daily_loss_usd=1.0))
    loser = bot.storage.add_trade(mode="paper", market_slug="old", market_end_ts=START, side="UP",
                                  token_id="x", price=0.7, size_usd=1.0, shares=1.4, fee_usd=0.02)
    bot.storage.settle(loser.id, won=False)
    asyncio.run(bot.step())
    assert bot.storage.stats("paper")["total"] == 1
    assert "лимит" in tg.sent[-1]


def test_bot_retries_rejected_orders_then_gives_up():
    class RejectingTrader(PaperTrader):
        attempts = 0

        async def buy(self, *args, **kwargs):
            RejectingTrader.attempts += 1
            raise OrderRejected("no match")

    clock = Clock(START + 790)
    bot, tg = make_bot(UP_STRONG, clock, trader=RejectingTrader(), settings=Settings(max_attempts_per_market=2))
    for _ in range(4):
        asyncio.run(bot.step())
        clock.t += 5
    assert RejectingTrader.attempts == 2
    assert bot.storage.stats("paper")["total"] == 0
    assert "попыток: 2" in tg.sent[-1]


def test_status_and_commands():
    bot, _ = make_bot(UP_STRONG, Clock(START + 100))
    asyncio.run(bot.step())
    status = asyncio.run(bot.handle_command("status", "", 1))
    assert "PAPER" in status and "Окно входа через 11:20" in status
    assert "Пауза" in asyncio.run(bot.handle_command("pause", "", 1)) and bot.paused
    asyncio.run(bot.handle_command("resume", "", 1))
    assert not bot.paused
    assert "Сделок пока нет" in asyncio.run(bot.handle_command("stats", "", 1))
    assert "купить UP" in asyncio.run(bot.handle_command("signals", "", 1))


# --- telegram ----------------------------------------------------------------------------------


def test_telegram_routes_commands_and_rejects_strangers():
    calls = []

    def handler(request: httpx.Request):
        method = request.url.path.rsplit("/", 1)[-1]
        body = json.loads(request.content or b"{}")
        calls.append((method, body))
        if method == "getUpdates":
            return httpx.Response(200, json={"ok": True, "result": [
                {"update_id": 1, "message": {"text": "/status@PolyBot now", "chat": {"id": 42}}},
                {"update_id": 2, "message": {"text": "/pause", "chat": {"id": 666}}},
                {"update_id": 3, "message": {"text": "hello", "chat": {"id": 42}}},
            ]})
        return httpx.Response(200, json={"ok": True, "result": {}})

    received = []

    async def on_command(command, arg, chat_id):
        received.append((command, arg, chat_id))
        return "ok"

    async def main():
        tg = Telegram("123:SECRET", (42,), http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        await tg.poll_once(on_command)
        await tg.aclose()
        return tg

    tg = asyncio.run(main())
    assert received == [("status", "now", 42)]  # stranger's /pause never reaches the bot
    sends = [b for m, b in calls if m == "sendMessage"]
    assert {"chat_id": 42, "text": "ok"}.items() <= sends[0].items()
    assert sends[1]["chat_id"] == 666 and "666" in sends[1]["text"]
    assert tg._offset == 4


def test_telegram_errors_do_not_leak_token():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    async def main():
        tg = Telegram("123:SECRET", (42,), http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        try:
            await tg.get_me()
        finally:
            await tg.aclose()

    with pytest.raises(RuntimeError) as exc:
        asyncio.run(main())
    assert "SECRET" not in str(exc.value)


def test_settings_validation():
    assert Settings().errors() == []
    assert any("POLYMARKET_PK" in e for e in Settings(live=True).errors())
    assert Settings(live=True, private_key="0x1", signature_type=0).errors() == []
    assert Settings(trade_size_usd=0.5).errors()
    assert Settings(signal_filter="maybe").errors()


# --- live trader (py-clob-client mocked, no network) ------------------------------------------


def make_live_trader():
    pytest.importorskip("py_clob_client")
    from polybot.trader import LiveTrader

    settings = Settings(live=True, private_key="0x" + "11" * 32, signature_type=0,
                        api_key="k", api_secret="s", api_passphrase="p")
    return LiveTrader(settings)


def test_live_trader_parses_fill_and_caps_price():
    trader = make_live_trader()
    seen = {}

    def create_market_order(args):
        seen["args"] = args
        return "signed-order"

    def post_order(order, order_type):
        seen["post"] = (order, order_type)
        return {"success": True, "orderID": "0xorder", "makingAmount": "1.0", "takingAmount": "1.38"}

    trader._client.create_market_order = create_market_order
    trader._client.post_order = post_order
    up = UP_STRONG["tok-up"]
    fill = asyncio.run(trader.buy("tok-up", 1.0, up, max_price=0.747, tick_size=0.01))

    assert seen["args"].price == 0.74 and seen["args"].amount == 1.0 and seen["args"].side == "BUY"
    assert seen["post"] == ("signed-order", "FOK")
    assert fill.order_id == "0xorder"
    assert fill.shares == pytest.approx(1.38) and fill.avg_price == pytest.approx(1.0 / 1.38)


def test_live_trader_classifies_errors():
    from py_clob_client.exceptions import PolyApiException
    from polybot.trader import OrderStateUnknown

    trader = make_live_trader()
    trader._client.create_market_order = lambda args: "signed"
    up = UP_STRONG["tok-up"]

    def raise_(exc):
        def post_order(order, order_type):
            raise exc
        return post_order

    rejected = PolyApiException(resp=httpx.Response(400, json={"error": "no match"}))
    trader._client.post_order = raise_(rejected)
    with pytest.raises(OrderRejected):
        asyncio.run(trader.buy("tok-up", 1.0, up, 0.8, 0.01))

    trader._client.post_order = lambda order, order_type: {"success": False, "errorMsg": "FOK not filled"}
    with pytest.raises(OrderRejected):
        asyncio.run(trader.buy("tok-up", 1.0, up, 0.8, 0.01))

    # Network failure or 5xx: the order may have gone through -> must not be retried.
    for exc in (PolyApiException(error_msg="Request exception!"),
                PolyApiException(resp=httpx.Response(502, text="bad gateway")),
                TimeoutError("read timeout")):
        trader._client.post_order = raise_(exc)
        with pytest.raises(OrderStateUnknown):
            asyncio.run(trader.buy("tok-up", 1.0, up, 0.8, 0.01))


def test_bot_does_not_retry_when_order_state_unknown():
    from polybot.trader import OrderStateUnknown

    class FlakyTrader(PaperTrader):
        attempts = 0

        async def buy(self, *args, **kwargs):
            FlakyTrader.attempts += 1
            raise OrderStateUnknown("timeout")

    clock = Clock(START + 790)
    bot, tg = make_bot(UP_STRONG, clock, trader=FlakyTrader())
    for _ in range(3):
        asyncio.run(bot.step())
        clock.t += 5
    assert FlakyTrader.attempts == 1
    assert "неизвестен" in tg.sent[-1]
