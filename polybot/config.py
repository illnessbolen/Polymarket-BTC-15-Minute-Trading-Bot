"""
Bot settings, read from environment variables (a local .env file is loaded too).

Every value has a safe default, so the bot runs in paper mode with nothing
configured. See .env.example for the full list with explanations.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SIGNAL_FILTERS = ("none", "any", "agree")


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _float(name: str, default: float) -> float:
    raw = _get(name, str(default))
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be a number, got {raw!r}") from e


def _int(name: str, default: int) -> int:
    raw = _get(name, str(default))
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from e


def _bool(name: str, default: bool) -> bool:
    raw = _get(name, "1" if default else "0").lower()
    return raw in ("1", "true", "yes", "on")


def _chat_ids(name: str) -> tuple[int, ...]:
    raw = _get(name, "")
    ids = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                ids.append(int(part))
            except ValueError as e:
                raise ValueError(f"{name} must be a comma-separated list of chat ids, got {part!r}") from e
    return tuple(ids)


@dataclass(frozen=True)
class Settings:
    # --- Mode -----------------------------------------------------------------
    live: bool = False

    # --- Strategy (defaults mirror bot.py) -------------------------------------
    trade_size_usd: float = 1.0
    trend_up_threshold: float = 0.60     # Up price above this -> buy Up
    trend_down_threshold: float = 0.40   # Up price below this -> buy Down
    max_entry_price: float = 0.95        # never pay more than this per share
    trade_window_start: int = 780        # seconds into the 15-min market (minute 13)
    trade_window_end: int = 840          # ... until minute 14
    signal_filter: str = "any"           # none | any | agree (see strategy.decide)
    min_fusion_score: float = 40.0
    max_attempts_per_market: int = 3

    # --- Risk -----------------------------------------------------------------
    max_daily_loss_usd: float = 10.0     # 0 disables the limit
    slippage: float = 0.02               # live: worst price = best ask + slippage

    # --- Paper trading ----------------------------------------------------------
    paper_fee_rate: float = 0.07         # Polymarket crypto taker fee rate

    # --- Runtime ----------------------------------------------------------------
    poll_interval: float = 5.0
    resolve_interval: float = 30.0
    db_path: Path = Path("data/polybot.db")
    log_dir: Path = Path("logs")
    gamma_url: str = "https://gamma-api.polymarket.com"
    clob_url: str = "https://clob.polymarket.com"

    # --- Telegram ---------------------------------------------------------------
    telegram_token: Optional[str] = None
    telegram_chat_ids: tuple[int, ...] = field(default_factory=tuple)
    notify_skips: bool = False

    # --- Polymarket credentials (live mode only) --------------------------------
    private_key: Optional[str] = None
    funder: Optional[str] = None
    signature_type: int = 1
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_passphrase: Optional[str] = None

    @classmethod
    def from_env(cls, live: bool = False) -> "Settings":
        return cls(
            live=live,
            trade_size_usd=_float("TRADE_SIZE_USD", cls.trade_size_usd),
            trend_up_threshold=_float("TREND_UP_THRESHOLD", cls.trend_up_threshold),
            trend_down_threshold=_float("TREND_DOWN_THRESHOLD", cls.trend_down_threshold),
            max_entry_price=_float("MAX_ENTRY_PRICE", cls.max_entry_price),
            trade_window_start=_int("TRADE_WINDOW_START", cls.trade_window_start),
            trade_window_end=_int("TRADE_WINDOW_END", cls.trade_window_end),
            signal_filter=_get("SIGNAL_FILTER", cls.signal_filter).lower(),
            min_fusion_score=_float("MIN_FUSION_SCORE", cls.min_fusion_score),
            max_attempts_per_market=_int("MAX_ATTEMPTS_PER_MARKET", cls.max_attempts_per_market),
            max_daily_loss_usd=_float("MAX_DAILY_LOSS_USD", cls.max_daily_loss_usd),
            slippage=_float("SLIPPAGE", cls.slippage),
            paper_fee_rate=_float("PAPER_FEE_RATE", cls.paper_fee_rate),
            poll_interval=_float("POLL_INTERVAL", cls.poll_interval),
            resolve_interval=_float("RESOLVE_INTERVAL", cls.resolve_interval),
            db_path=Path(_get("DB_PATH", str(cls.db_path))),
            log_dir=Path(_get("LOG_DIR", str(cls.log_dir))),
            gamma_url=_get("GAMMA_URL", cls.gamma_url).rstrip("/"),
            clob_url=_get("CLOB_URL", cls.clob_url).rstrip("/"),
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
            telegram_chat_ids=_chat_ids("TELEGRAM_CHAT_ID"),
            notify_skips=_bool("NOTIFY_SKIPS", cls.notify_skips),
            private_key=os.getenv("POLYMARKET_PK") or None,
            funder=os.getenv("POLYMARKET_FUNDER") or None,
            signature_type=_int("POLYMARKET_SIGNATURE_TYPE", cls.signature_type),
            api_key=os.getenv("POLYMARKET_API_KEY") or None,
            api_secret=os.getenv("POLYMARKET_API_SECRET") or None,
            api_passphrase=os.getenv("POLYMARKET_PASSPHRASE") or None,
        )

    @property
    def mode(self) -> str:
        return "live" if self.live else "paper"

    def errors(self) -> list[str]:
        """Return a list of configuration problems (empty when everything is fine)."""
        errs = []
        if self.trade_size_usd < 1.0:
            errs.append("TRADE_SIZE_USD must be at least 1.0 (Polymarket minimum for market buys)")
        if not 0.5 <= self.trend_up_threshold < 1.0:
            errs.append("TREND_UP_THRESHOLD must be in [0.5, 1.0)")
        if not 0.0 < self.trend_down_threshold <= 0.5:
            errs.append("TREND_DOWN_THRESHOLD must be in (0.0, 0.5]")
        if not 0.0 < self.max_entry_price < 1.0:
            errs.append("MAX_ENTRY_PRICE must be in (0.0, 1.0)")
        if not 0 <= self.trade_window_start < self.trade_window_end <= 900:
            errs.append("TRADE_WINDOW_START/END must satisfy 0 <= start < end <= 900")
        if self.signal_filter not in SIGNAL_FILTERS:
            errs.append(f"SIGNAL_FILTER must be one of {', '.join(SIGNAL_FILTERS)}")
        if self.max_attempts_per_market < 1:
            errs.append("MAX_ATTEMPTS_PER_MARKET must be at least 1")
        if self.poll_interval <= 0 or self.resolve_interval <= 0:
            errs.append("POLL_INTERVAL and RESOLVE_INTERVAL must be positive")
        if self.live:
            if not self.private_key:
                errs.append("POLYMARKET_PK is required for live trading")
            if self.signature_type not in (0, 1, 2):
                errs.append("POLYMARKET_SIGNATURE_TYPE must be 0 (EOA), 1 (email/Magic) or 2 (browser wallet)")
            if self.signature_type != 0 and not self.funder:
                errs.append("POLYMARKET_FUNDER (your Polymarket proxy wallet address) is required "
                            "when POLYMARKET_SIGNATURE_TYPE is 1 or 2")
        return errs
