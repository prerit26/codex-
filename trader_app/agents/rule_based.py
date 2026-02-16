from __future__ import annotations

from typing import Dict, List

from trader_app.config import AppConfig
from trader_app.types import AgentContext, TradeDecision


POSITIVE_NEWS_TERMS = {
    "beat",
    "growth",
    "upgrade",
    "surge",
    "profit",
    "strong",
    "record",
    "bullish",
    "expansion",
}
NEGATIVE_NEWS_TERMS = {
    "miss",
    "downgrade",
    "fall",
    "loss",
    "weak",
    "fraud",
    "probe",
    "bearish",
    "cut",
    "decline",
}


def _headline_score(title: str) -> float:
    text = (title or "").lower()
    if not text:
        return 0.0
    pos = sum(1 for token in POSITIVE_NEWS_TERMS if token in text)
    neg = sum(1 for token in NEGATIVE_NEWS_TERMS if token in text)
    return float(pos - neg)


class RuleBasedAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _held_symbols(self, context: AgentContext) -> set[str]:
        held = set()
        for position in context.portfolio.get("positions", []):
            symbol = str(position.get("symbol", "")).upper()
            if symbol:
                held.add(symbol)
        return held

    def _symbol_score(self, symbol: str, snapshot: Dict, news_rows: List[Dict]) -> float:
        ret_20 = float(snapshot.get("return_20d", 0.0))
        ret_60 = float(snapshot.get("return_60d", 0.0))
        ret_120 = float(snapshot.get("return_120d", 0.0))
        vol = max(0.01, float(snapshot.get("volatility_20d", 0.0)))

        sma_20 = float(snapshot.get("sma_20", 0.0))
        sma_50 = float(snapshot.get("sma_50", 0.0))
        sma_100 = float(snapshot.get("sma_100", 0.0))
        last_close = float(snapshot.get("last_close", 0.0))

        trend_bonus = 0.0
        if last_close > sma_20 > sma_50 > sma_100 > 0:
            trend_bonus = 0.05
        elif last_close < sma_20 < sma_50:
            trend_bonus = -0.05

        momentum = 0.45 * ret_20 + 0.35 * ret_60 + 0.20 * ret_120
        risk_adjusted = momentum / vol

        news_score = 0.0
        if news_rows:
            title_scores = [_headline_score(str(item.get("title", ""))) for item in news_rows]
            news_score = sum(title_scores) / max(1.0, len(title_scores))
            news_score = max(-1.5, min(1.5, news_score)) * 0.02

        return risk_adjusted + trend_bonus + news_score

    def _macro_risk_factor(self, context: AgentContext) -> float:
        macro = context.macro_state or {}
        vix_row = macro.get("_INDIAVIX", {}) or macro.get("^INDIAVIX", {})
        inr_row = macro.get("INRUSD=X", {})
        crude_row = macro.get("CL=F", {})

        penalty = 0.0
        vix_level = float(vix_row.get("last_close", 0.0)) if vix_row else 0.0
        if vix_level > 24:
            penalty -= 0.15
        elif vix_level > 18:
            penalty -= 0.06

        if inr_row:
            inr_ret20 = float(inr_row.get("return_20d", 0.0))
            penalty += -0.03 if inr_ret20 > 0.02 else 0.0

        if crude_row:
            crude_ret20 = float(crude_row.get("return_20d", 0.0))
            penalty += -0.03 if crude_ret20 > 0.06 else 0.0

        return penalty

    def _fundamental_tilt(self, fundamentals: Dict) -> float:
        if not fundamentals:
            return 0.0
        tilt = 0.0
        pe = fundamentals.get("trailingPE")
        roe = fundamentals.get("returnOnEquity")
        rev_g = fundamentals.get("revenueGrowth")
        margin = fundamentals.get("profitMargins")
        debt = fundamentals.get("debtToEquity")

        try:
            if pe is not None and 0 < float(pe) < 35:
                tilt += 0.03
            if roe is not None and float(roe) > 0.12:
                tilt += 0.03
            if rev_g is not None and float(rev_g) > 0.05:
                tilt += 0.02
            if margin is not None and float(margin) > 0.08:
                tilt += 0.02
            if debt is not None and float(debt) > 1.5:
                tilt -= 0.03
        except Exception:
            return 0.0
        return tilt

    def decide(self, context: AgentContext) -> List[TradeDecision]:
        if not context.market_state:
            return []

        held_symbols = self._held_symbols(context)
        macro_factor = self._macro_risk_factor(context)
        scored: List[Dict] = []
        for symbol, snapshot in context.market_state.items():
            score = self._symbol_score(symbol, snapshot, context.news_state.get(symbol, []))
            score += macro_factor
            score += self._fundamental_tilt(context.fundamental_state.get(symbol, {}))
            scored.append({"symbol": symbol, "score": score, "snapshot": snapshot})

        scored.sort(key=lambda item: item["score"], reverse=True)
        if not scored:
            return []

        best_score = scored[0]["score"]
        worst_score = scored[-1]["score"]
        score_span = max(1e-9, best_score - worst_score)

        buy_candidates = [item for item in scored if item["score"] > 0]
        sell_candidates = [item for item in scored if item["score"] < 0]

        decisions: List[TradeDecision] = []

        max_new = self.config.agent.max_new_positions_per_step
        for item in buy_candidates[: max_new * 2]:
            symbol = item["symbol"]
            score = item["score"]
            rel = (score - worst_score) / score_span
            confidence = 0.45 + 0.5 * rel
            confidence = max(0.0, min(1.0, confidence))

            vol = max(0.01, float(item["snapshot"].get("volatility_20d", 0.2)))
            vol_scale = max(0.55, min(1.4, 0.22 / vol))
            size_pct = self.config.agent.default_position_pct * vol_scale * (0.8 + 0.6 * rel)
            size_pct = max(0.02, min(self.config.agent.max_position_pct, size_pct))

            decisions.append(
                TradeDecision(
                    symbol=symbol,
                    action="BUY",
                    confidence=confidence,
                    size_pct=size_pct,
                    rationale="Momentum+trend+news composite score is positive.",
                )
            )

        for item in sell_candidates:
            symbol = item["symbol"]
            if symbol not in held_symbols:
                continue
            score = item["score"]
            confidence = 0.55 + min(0.35, abs(score) * 0.2)
            size_pct = 1.0 if score < -0.08 else 0.5
            decisions.append(
                TradeDecision(
                    symbol=symbol,
                    action="SELL",
                    confidence=max(0.0, min(1.0, confidence)),
                    size_pct=size_pct,
                    rationale="Composite score turned negative for held position.",
                )
            )

        return decisions
