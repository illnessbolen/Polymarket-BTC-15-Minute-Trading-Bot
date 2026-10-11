"""
Polybot — standalone Polymarket BTC 15-minute trading bot with Telegram control.

Reuses the signal processors and fusion engine from ``core/strategy_brain``
but talks to Polymarket directly (Gamma + CLOB HTTP APIs) instead of going
through NautilusTrader and its monkey patches.
"""

__version__ = "1.0.0"
