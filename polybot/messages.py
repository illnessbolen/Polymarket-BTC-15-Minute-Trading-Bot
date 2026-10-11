"""Telegram message formatting (HTML parse mode). Pure functions, no I/O."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from typing import Optional

from polybot.storage import Trade

SIDE_LABEL = {"UP": "UP ⬆️", "DOWN": "DOWN ⬇️"}
MODE_LABEL = {"paper": "📝 PAPER", "live": "💰 LIVE"}

HELP = (
    "<b>Polybot</b> — бот для 15-минутных рынков BTC Up/Down на Polymarket\n\n"
    "/status — текущий рынок, цены и состояние бота\n"
    "/signals — что показывают сигналы прямо сейчас\n"
    "/stats — статистика сделок\n"
    "/trades — последние 10 сделок\n"
    "/pause — приостановить открытие новых сделок\n"
    "/resume — продолжить торговлю\n"
    "/help — эта справка"
)


def usd(value: Optional[float]) -> str:
    if value is None:
        return "—"
    sign = "+" if value > 0 else "−" if value < 0 else ""
    return f"{sign}${abs(value):.2f}"


def mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.0%}"


def signals_block(summary: dict) -> str:
    lines = []
    for s in summary.get("signals", []):
        arrow = "🟢" if s["direction"] == "BULLISH" else "🔴"
        lines.append(f"  {arrow} {escape(s['source'])}: {s['direction']} (score {s['score']:.0f})")
    if not lines:
        lines.append("  — сигналов нет")
    fused = summary.get("fused")
    if fused:
        lines.append(f"  Консенсус: <b>{fused['direction']}</b> (score {fused['score']:.0f})")
    return "\n".join(lines)


def startup(mode: str, size: float, settings_line: str, paused: bool) -> str:
    text = (
        f"🚀 <b>Polybot запущен</b> — {MODE_LABEL[mode]}\n"
        f"Ставка: ${size:.2f} на рынок\n"
        f"{escape(settings_line)}"
    )
    if paused:
        text += "\n⏸ Бот на паузе — /resume чтобы начать торговлю"
    return text


def trade_opened(trade: Trade, question: str, reason: str) -> str:
    return (
        f"🟢 <b>Сделка открыта</b> — {MODE_LABEL[trade.mode]}\n"
        f"{escape(question)}\n"
        f"Ставка: <b>{SIDE_LABEL[trade.side]}</b> по {trade.price:.3f} × {trade.shares:.2f} акций "
        f"= ${trade.size_usd:.2f} (комиссия ≈ ${trade.fee_usd:.3f})\n"
        f"Причина: {escape(reason)}\n"
        f"Сигналы:\n{signals_block(trade.signals)}"
    )


def trade_resolved(trade: Trade, winner: str, totals: dict) -> str:
    head = "✅ <b>Выигрыш</b>" if trade.status == "WON" else "❌ <b>Проигрыш</b>"
    return (
        f"{head} {usd(trade.pnl_usd)} — {MODE_LABEL[trade.mode]}\n"
        f"{escape(trade.market_slug)}: ставка {SIDE_LABEL[trade.side]} по {trade.price:.3f}, "
        f"итог {SIDE_LABEL[winner]}\n"
        f"Всего: {usd(totals['pnl'])} | побед {totals['won']}/{totals['won'] + totals['lost']} "
        f"({pct(totals['win_rate'])})"
    )


def skipped(question: str, reason: str) -> str:
    return f"⏭ Пропуск: {escape(question)}\n{escape(reason)}"


def stats(by_mode: dict[str, dict], today: dict[str, dict]) -> str:
    if not any(s["total"] for s in by_mode.values()):
        return "📊 Сделок пока нет."
    parts = ["📊 <b>Статистика</b>"]
    for mode, s in by_mode.items():
        if not s["total"]:
            continue
        t = today[mode]
        parts.append(
            f"\n{MODE_LABEL[mode]}\n"
            f"Сделок: {s['total']} (открыто {s['open']})\n"
            f"Побед: {s['won']}, поражений: {s['lost']}, win rate {pct(s['win_rate'])}\n"
            f"P&L: <b>{usd(s['pnl'])}</b> (комиссии ${s['fees']:.2f}, оборот ${s['volume']:.2f})\n"
            f"Сегодня: {usd(t['pnl'])}, сделок {t['total']}"
        )
    return "\n".join(parts)


def trades(items: list[Trade]) -> str:
    if not items:
        return "Сделок пока нет."
    lines = ["🧾 <b>Последние сделки</b>"]
    for t in items:
        when = datetime.fromisoformat(t.created_at).astimezone(timezone.utc).strftime("%m-%d %H:%M")
        if t.status == "OPEN":
            result = "⏳ ждёт итога"
        else:
            result = ("✅ " if t.status == "WON" else "❌ ") + usd(t.pnl_usd)
        lines.append(
            f"#{t.id} {when} {t.mode.upper()} {SIDE_LABEL[t.side]} @{t.price:.2f} ${t.size_usd:.2f} → {result}"
        )
    return "\n".join(lines)
