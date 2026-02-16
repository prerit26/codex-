from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class TradeDecision:
    symbol: str
    action: str
    confidence: float
    size_pct: float
    rationale: str = ""


@dataclass
class Order:
    timestamp: datetime
    symbol: str
    side: str
    quantity: int
    rationale: str = ""


@dataclass
class Fill:
    timestamp: datetime
    symbol: str
    side: str
    quantity: int
    price: float
    notional: float
    commission: float
    slippage_bps: float
    rationale: str = ""


@dataclass
class Position:
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0


@dataclass
class AgentContext:
    timestamp: datetime
    symbols: List[str]
    portfolio: Dict[str, Any]
    market_state: Dict[str, Any]
    news_state: Dict[str, Any]
    macro_state: Dict[str, Any]
    fundamental_state: Dict[str, Any]
