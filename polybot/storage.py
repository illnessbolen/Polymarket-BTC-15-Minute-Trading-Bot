"""
SQLite trade journal plus a tiny key/value table for bot state (e.g. pause).

One trade per market and mode is enforced by a unique index, so a restart in
the middle of a trade window can never place a second order on the same market.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    mode          TEXT    NOT NULL,
    market_slug   TEXT    NOT NULL,
    market_end_ts INTEGER NOT NULL,
    side          TEXT    NOT NULL,
    token_id      TEXT    NOT NULL,
    price         REAL    NOT NULL,
    size_usd      REAL    NOT NULL,
    shares        REAL    NOT NULL,
    fee_usd       REAL    NOT NULL DEFAULT 0,
    order_id      TEXT,
    signals       TEXT,
    status        TEXT    NOT NULL DEFAULT 'OPEN',
    pnl_usd       REAL,
    resolved_at   TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS trades_market_mode ON trades (market_slug, mode);
CREATE INDEX IF NOT EXISTS trades_status ON trades (status);
CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Trade:
    id: int
    created_at: str
    mode: str
    market_slug: str
    market_end_ts: int
    side: str
    token_id: str
    price: float
    size_usd: float
    shares: float
    fee_usd: float
    order_id: Optional[str]
    signals: dict
    status: str
    pnl_usd: Optional[float]
    resolved_at: Optional[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Trade":
        data = dict(row)
        data["signals"] = json.loads(data["signals"]) if data["signals"] else {}
        return cls(**data)


class Storage:
    def __init__(self, path: Path | str):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # --- trades -----------------------------------------------------------------

    def add_trade(
        self,
        *,
        mode: str,
        market_slug: str,
        market_end_ts: int,
        side: str,
        token_id: str,
        price: float,
        size_usd: float,
        shares: float,
        fee_usd: float = 0.0,
        order_id: Optional[str] = None,
        signals: Optional[dict] = None,
    ) -> Trade:
        cur = self._db.execute(
            """INSERT INTO trades (created_at, mode, market_slug, market_end_ts, side, token_id,
                                   price, size_usd, shares, fee_usd, order_id, signals)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (_now_iso(), mode, market_slug, market_end_ts, side, token_id, price,
             size_usd, shares, fee_usd, order_id, json.dumps(signals or {})),
        )
        self._db.commit()
        return self.get_trade(cur.lastrowid)

    def get_trade(self, trade_id: int) -> Trade:
        row = self._db.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        if row is None:
            raise KeyError(trade_id)
        return Trade.from_row(row)

    def has_trade(self, market_slug: str, mode: str) -> bool:
        row = self._db.execute(
            "SELECT 1 FROM trades WHERE market_slug = ? AND mode = ?", (market_slug, mode)
        ).fetchone()
        return row is not None

    def open_trades(self) -> list[Trade]:
        rows = self._db.execute(
            "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY id"
        ).fetchall()
        return [Trade.from_row(r) for r in rows]

    def settle(self, trade_id: int, won: bool) -> Trade:
        trade = self.get_trade(trade_id)
        payout = trade.shares if won else 0.0
        pnl = payout - trade.size_usd - trade.fee_usd
        self._db.execute(
            "UPDATE trades SET status = ?, pnl_usd = ?, resolved_at = ? WHERE id = ?",
            ("WON" if won else "LOST", pnl, _now_iso(), trade_id),
        )
        self._db.commit()
        return self.get_trade(trade_id)

    def recent(self, limit: int = 10, mode: Optional[str] = None) -> list[Trade]:
        if mode:
            rows = self._db.execute(
                "SELECT * FROM trades WHERE mode = ? ORDER BY id DESC LIMIT ?", (mode, limit)
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [Trade.from_row(r) for r in rows]

    def stats(self, mode: str, since: Optional[str] = None) -> dict:
        """Aggregate results for ``mode``; ``since`` is an ISO timestamp lower bound."""
        where, args = "mode = ?", [mode]
        if since:
            where += " AND created_at >= ?"
            args.append(since)
        row = self._db.execute(
            f"""SELECT COUNT(*)                                      AS total,
                       SUM(status = 'WON')                           AS won,
                       SUM(status = 'LOST')                          AS lost,
                       SUM(status = 'OPEN')                          AS open,
                       COALESCE(SUM(pnl_usd), 0)                     AS pnl,
                       COALESCE(SUM(fee_usd), 0)                     AS fees,
                       COALESCE(SUM(size_usd), 0)                    AS volume,
                       COALESCE(SUM(CASE WHEN status = 'OPEN' THEN size_usd END), 0) AS at_risk
                FROM trades WHERE {where}""",
            args,
        ).fetchone()
        won, lost = row["won"] or 0, row["lost"] or 0
        return {
            "total": row["total"],
            "won": won,
            "lost": lost,
            "open": row["open"] or 0,
            "pnl": row["pnl"],
            "fees": row["fees"],
            "volume": row["volume"],
            "at_risk": row["at_risk"],
            "win_rate": won / (won + lost) if won + lost else None,
        }

    # --- state ------------------------------------------------------------------

    def get_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self._db.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT INTO state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._db.commit()
