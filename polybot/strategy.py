"""
Trading decision: the late-window trend filter from bot.py, gated by the
repository's signal processors and fusion engine (core/strategy_brain).

At minute 13 of a 15-minute market the Up price already reflects where BTC
has gone during the interval, so the bot follows it:

    Up price > TREND_UP_THRESHOLD    -> buy Up
    Up price < TREND_DOWN_THRESHOLD  -> buy Down
    otherwise                        -> skip (coin-flip territory)

The six signal processors still run and are fused into one consensus.
SIGNAL_FILTER controls how much that consensus matters:

    none   trend filter only
    any    trade only if fusion produced a signal at all (bot.py behaviour)
    agree  trade only if the fused direction agrees with the trend
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from loguru import logger

from core.strategy_brain.fusion_engine.signal_fusion import FusedSignal, SignalFusionEngine
from core.strategy_brain.signal_processors.base_processor import TradingSignal
from core.strategy_brain.signal_processors.deribit_pcr_processor import DeribitPCRProcessor
from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
from core.strategy_brain.signal_processors.orderbook_processor import OrderBookImbalanceProcessor
from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor

from polybot.config import Settings

# Same weights as bot.py (higher = more influence on the fused direction).
FUSION_WEIGHTS = {
    "OrderBookImbalance": 0.30,
    "TickVelocity": 0.25,
    "PriceDivergence": 0.18,
    "SpikeDetection": 0.12,
    "DeribitPCR": 0.10,
    "SentimentAnalysis": 0.05,
}


def is_bullish(direction) -> bool:
    return "BULLISH" in str(direction).upper()


@dataclass
class Snapshot:
    """Everything the processors need, gathered by the engine."""
    up_price: float
    price_history: list[float]
    tick_buffer: list[dict]          # [{'ts': aware datetime, 'price': float}, ...]
    up_token_id: str
    spot_price: Optional[float] = None
    fear_greed: Optional[dict] = None


@dataclass
class Evaluation:
    signals: list[TradingSignal] = field(default_factory=list)
    fused: Optional[FusedSignal] = None

    def summary(self) -> dict:
        """JSON-friendly summary for the trade journal and Telegram."""
        out = {
            "signals": [
                {
                    "source": s.source,
                    "direction": "BULLISH" if is_bullish(s.direction) else "BEARISH",
                    "score": round(s.score, 1),
                    "confidence": round(s.confidence, 3),
                }
                for s in self.signals
            ],
            "fused": None,
        }
        if self.fused:
            out["fused"] = {
                "direction": "BULLISH" if is_bullish(self.fused.direction) else "BEARISH",
                "score": round(self.fused.score, 1),
                "confidence": round(self.fused.confidence, 3),
            }
        return out


class SignalEngine:
    """The six processors from bot.py, configured identically, plus fusion."""

    def __init__(self, min_fusion_score: float = 40.0):
        self.min_fusion_score = min_fusion_score
        self.processors = [
            SpikeDetectionProcessor(spike_threshold=0.05, lookback_periods=20),
            SentimentProcessor(extreme_fear_threshold=25, extreme_greed_threshold=75),
            PriceDivergenceProcessor(divergence_threshold=0.05),
            OrderBookImbalanceProcessor(imbalance_threshold=0.30, min_book_volume=50.0),
            TickVelocityProcessor(velocity_threshold_60s=0.015, velocity_threshold_30s=0.010),
            DeribitPCRProcessor(bullish_pcr_threshold=1.20, bearish_pcr_threshold=0.70,
                                max_days_to_expiry=2, cache_seconds=300),
        ]
        self.fusion = SignalFusionEngine()
        for name, weight in FUSION_WEIGHTS.items():
            self.fusion.set_weight(name, weight)

    @staticmethod
    def build_metadata(snap: Snapshot) -> dict:
        history = snap.price_history or [snap.up_price]
        recent = history[-20:]
        sma = sum(recent) / len(recent)
        deviation = (snap.up_price - sma) / sma if sma else 0.0
        momentum = (snap.up_price - history[-5]) / history[-5] if len(history) >= 5 and history[-5] else 0.0
        volatility = math.sqrt(sum((p - sma) ** 2 for p in recent) / len(recent))

        metadata = {
            "deviation": deviation,
            "momentum": momentum,
            "volatility": volatility,
            "tick_buffer": list(snap.tick_buffer),
            "yes_token_id": snap.up_token_id,
        }
        if snap.spot_price is not None:
            metadata["spot_price"] = snap.spot_price
        if snap.fear_greed is not None:
            metadata["sentiment_score"] = snap.fear_greed["value"]
            metadata["sentiment_classification"] = snap.fear_greed.get("classification", "")
        return metadata

    def evaluate(self, snap: Snapshot) -> Evaluation:
        """Blocking (some processors do HTTP) — call it from a worker thread."""
        current = Decimal(str(snap.up_price))
        history = [Decimal(str(p)) for p in snap.price_history]
        metadata = self.build_metadata(snap)

        signals = []
        for processor in self.processors:
            try:
                signal = processor.process(current_price=current, historical_prices=history, metadata=metadata)
            except Exception as e:  # one broken processor must not stop trading
                logger.warning(f"{processor.name} failed: {e}")
                continue
            if signal:
                signals.append(signal)

        fused = None
        if signals:
            fused = self.fusion.fuse_signals(signals, min_signals=1, min_score=self.min_fusion_score)
        return Evaluation(signals=signals, fused=fused)


@dataclass(frozen=True)
class Decision:
    side: Optional[str]  # "UP", "DOWN" or None to skip
    reason: str


def decide(up_price: float, evaluation: Evaluation, settings: Settings) -> Decision:
    if up_price > settings.trend_up_threshold:
        side = "UP"
    elif up_price < settings.trend_down_threshold:
        side = "DOWN"
    else:
        return Decision(None, f"цена Up {up_price:.2f} в нейтральной зоне "
                              f"{settings.trend_down_threshold:.2f}–{settings.trend_up_threshold:.2f}")

    fused = evaluation.fused
    if settings.signal_filter == "any" and fused is None:
        return Decision(None, "нет сигналов от процессоров (SIGNAL_FILTER=any)")
    if settings.signal_filter == "agree":
        if fused is None:
            return Decision(None, "нет сигналов от процессоров (SIGNAL_FILTER=agree)")
        if is_bullish(fused.direction) != (side == "UP"):
            return Decision(None, f"сигналы против тренда: тренд {side}, "
                                  f"консенсус {'BULLISH' if is_bullish(fused.direction) else 'BEARISH'}")

    return Decision(side, f"тренд {side}: цена Up {up_price:.2f}")
