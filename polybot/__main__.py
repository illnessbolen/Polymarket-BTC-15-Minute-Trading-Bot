"""
Command line entry point.

    python -m polybot run           # paper trading (default)
    python -m polybot run --live    # real orders — real money!
    python -m polybot check         # verify APIs, Telegram and configuration
    python -m polybot stats         # print results from the trade journal
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import html
import logging
import re
import signal
import sys
import time

from dotenv import load_dotenv
from loguru import logger

from polybot import messages
from polybot.config import Settings
from polybot.engine import Bot, utc_midnight_iso
from polybot.market_data import MarketData
from polybot.polymarket import PolymarketClient, window_start
from polybot.storage import Storage
from polybot.strategy import SignalEngine
from polybot.telegram import Telegram
from polybot.trader import LiveTrader, PaperTrader


def plain(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def setup_logging(settings: Settings, level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(settings.log_dir / "polybot.log", level="INFO", rotation="10 MB", retention=5, encoding="utf-8")
    # httpx logs full request URLs at INFO — those include the Telegram token.
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def run_bot(settings: Settings) -> None:
    storage = Storage(settings.db_path)
    if settings.live:
        trader = LiveTrader(settings, fee_rate=settings.paper_fee_rate)
        balance = await trader.balance_usd()
        if balance is not None:
            logger.info(f"Polymarket USDC balance: ${balance:.2f}")
    else:
        trader = PaperTrader(fee_rate=settings.paper_fee_rate)

    polymarket = PolymarketClient(settings.gamma_url, settings.clob_url)
    telegram = Telegram(settings.telegram_token, settings.telegram_chat_ids)
    market_data = MarketData()
    bot = Bot(settings, polymarket, trader, storage, telegram, market_data, SignalEngine(settings.min_fusion_score))

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # Windows: Ctrl+C raises KeyboardInterrupt instead
            pass

    try:
        await bot.run(stop)
    finally:
        await asyncio.gather(polymarket.aclose(), telegram.aclose(), market_data.aclose(), return_exceptions=True)
        storage.close()


async def check(settings: Settings) -> int:
    ok = True
    polymarket = PolymarketClient(settings.gamma_url, settings.clob_url)
    market_data = MarketData()
    try:
        start = window_start(time.time())
        market = await polymarket.get_market(start)
        if market is None:
            print(f"✗ Polymarket: no market for window {start}")
            ok = False
        else:
            print(f"✓ Polymarket market: {market.question} ({market.slug})")
            for side in ("UP", "DOWN"):
                book = await polymarket.get_book(market.token_for(side))
                print(f"    {side:<4} bid {book.best_bid} / ask {book.best_ask}")
    except Exception as e:
        print(f"✗ Polymarket API error: {e!r}")
        ok = False

    spot = await market_data.spot_price()
    if spot:
        print(f"✓ Coinbase BTC spot: ${spot:,.2f}")
    else:
        print("– Coinbase BTC spot: unavailable (PriceDivergence falls back to Polymarket momentum)")
    fg = await market_data.fear_greed()
    if fg:
        print(f"✓ Fear & Greed: {fg['value']:.0f} ({fg['classification']})")
    else:
        print("– Fear & Greed: unavailable (sentiment signal skipped)")
    await asyncio.gather(polymarket.aclose(), market_data.aclose())

    if settings.telegram_token:
        telegram = Telegram(settings.telegram_token, settings.telegram_chat_ids)
        try:
            me = await telegram.get_me()
            print(f"✓ Telegram bot: @{me.get('username')}")
            if settings.telegram_chat_ids:
                print(f"    allowed chats: {', '.join(map(str, settings.telegram_chat_ids))}")
            else:
                print("    ! TELEGRAM_CHAT_ID is empty: start the bot, send it /start and it will reply with your chat id")
        except Exception as e:
            print(f"✗ Telegram: {e}")
            ok = False
        finally:
            await telegram.aclose()
    else:
        print("– Telegram: TELEGRAM_BOT_TOKEN not set (notifications disabled)")

    paper_errors = settings.errors()
    live_errors = dataclasses.replace(settings, live=True).errors()
    for err in paper_errors:
        print(f"✗ Config: {err}")
    ok = ok and not paper_errors
    if live_errors:
        print("– Live trading not configured: " + "; ".join(e for e in live_errors if e not in paper_errors))
    else:
        print("✓ Live trading credentials present")
    return 0 if ok else 1


def print_stats(settings: Settings) -> None:
    storage = Storage(settings.db_path)
    now = time.time()
    modes = ("paper", "live")
    print(plain(messages.stats(
        {m: storage.stats(m) for m in modes},
        {m: storage.stats(m, since=utc_midnight_iso(now)) for m in modes},
    )))
    print()
    print(plain(messages.trades(storage.recent(10))))
    storage.close()


def main(argv=None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="polybot", description="Polymarket BTC 15-minute trading bot")
    parser.add_argument("--log-level", default="INFO", help="console log level (default INFO)")
    sub = parser.add_subparsers(dest="command")
    run_parser = sub.add_parser("run", help="run the bot (paper trading unless --live)")
    run_parser.add_argument("--live", action="store_true", help="place REAL orders with real money")
    sub.add_parser("check", help="check API connectivity, Telegram and configuration")
    sub.add_parser("stats", help="print trade statistics")
    args = parser.parse_args(argv)
    command = args.command or "run"

    try:
        settings = Settings.from_env(live=getattr(args, "live", False))
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2

    if command == "check":
        return asyncio.run(check(settings))
    if command == "stats":
        print_stats(settings)
        return 0

    setup_logging(settings, args.log_level.upper())
    errors = settings.errors()
    if errors:
        for err in errors:
            logger.error(f"Configuration error: {err}")
        return 2
    if settings.live:
        logger.warning("=" * 70)
        logger.warning("LIVE MODE — real orders, real money. Ctrl+C to stop.")
        logger.warning("=" * 70)
    if settings.telegram_token and not settings.telegram_chat_ids:
        logger.warning("TELEGRAM_CHAT_ID is empty: send /start to your bot to learn your chat id")

    try:
        asyncio.run(run_bot(settings))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
