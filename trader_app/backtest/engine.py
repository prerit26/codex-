from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Protocol

import pandas as pd
from tqdm import tqdm

from trader_app.config import AppConfig
from trader_app.types import AgentContext, Order, TradeDecision
from .broker import SimBroker
from .feed import ReplayDataFeed
from .metrics import compute_backtest_metrics


class TradingAgentProtocol(Protocol):
    def decide(self, context: AgentContext) -> List[TradeDecision]:
        ...


@dataclass
class BacktestResult:
    summary: Dict
    equity_curve: pd.DataFrame
    fills: pd.DataFrame
    decisions: pd.DataFrame


class BacktestEngine:
    def __init__(
        self,
        config: AppConfig,
        feed: ReplayDataFeed,
        broker: SimBroker,
        agent: TradingAgentProtocol,
    ) -> None:
        self.config = config
        self.feed = feed
        self.broker = broker
        self.agent = agent
        self.decision_log: List[Dict] = []

    def _position_qty(self, symbol: str) -> int:
        position = self.broker.positions.get(symbol)
        return position.quantity if position else 0

    def _clamp_buy_size_pct(self, raw: float) -> float:
        value = float(raw) if raw is not None else self.config.agent.default_position_pct
        value = max(0.0, value)
        return min(value, self.config.agent.max_position_pct)

    def _can_add_buy_exposure(
        self,
        equity: float,
        reference_prices: Dict[str, float],
        symbol: str,
        qty: int,
    ) -> bool:
        if qty <= 0:
            return False
        price = reference_prices.get(symbol)
        if price is None:
            return False
        order_value = qty * price
        reserve_cash = equity * self.config.agent.min_cash_reserve_pct
        if (
            self.config.agent.min_cash_reserve_pct > 0
            and self.broker.cash - order_value < reserve_cash
        ):
            return False
        projected_gross = self.broker.gross_exposure(reference_prices) + order_value
        return projected_gross <= equity * self.config.backtest.max_gross_leverage

    def _decision_to_order(
        self,
        timestamp: pd.Timestamp,
        decision: TradeDecision,
        reference_prices: Dict[str, float],
        equity: float,
    ) -> Order | None:
        symbol = decision.symbol.upper()
        action = decision.action.upper()
        if symbol not in reference_prices:
            return None

        price = reference_prices[symbol]
        current_qty = self._position_qty(symbol)

        if action == "BUY":
            target_pct = self._clamp_buy_size_pct(decision.size_pct)
            target_qty = int((equity * target_pct) / price) if price > 0 else 0
            buy_qty = max(0, target_qty - current_qty)
            if buy_qty <= 0:
                return None
            if not self._can_add_buy_exposure(equity, reference_prices, symbol, buy_qty):
                return None
            return Order(
                timestamp=timestamp.to_pydatetime(),
                symbol=symbol,
                side="BUY",
                quantity=buy_qty,
                rationale=decision.rationale,
            )

        if action == "SELL":
            if current_qty <= 0:
                return None
            raw_sell = float(decision.size_pct) if decision.size_pct is not None else 1.0
            raw_sell = max(0.0, raw_sell)
            sell_fraction = min(1.0, max(0.05, raw_sell))
            sell_qty = max(1, int(current_qty * sell_fraction))
            return Order(
                timestamp=timestamp.to_pydatetime(),
                symbol=symbol,
                side="SELL",
                quantity=sell_qty,
                rationale=decision.rationale,
            )

        return None

    def _log_decision(
        self,
        ts: pd.Timestamp,
        decision: TradeDecision,
        accepted: bool,
        reason: str,
    ) -> None:
        self.decision_log.append(
            {
                "timestamp": ts,
                "symbol": decision.symbol,
                "action": decision.action,
                "confidence": decision.confidence,
                "size_pct": decision.size_pct,
                "accepted": accepted,
                "reason": reason,
                "rationale": decision.rationale,
            }
        )

    def _run_decision_session(
        self,
        ts: pd.Timestamp,
        reference_prices: Dict[str, float],
        market_include_current: bool,
        queue_reason: str,
    ) -> None:
        symbols = self.feed.available_symbols()[: self.config.agent.max_symbols_per_step]
        portfolio_state = self.broker.portfolio_snapshot(reference_prices)
        equity_ref = float(portfolio_state.get("equity", self.broker.cash))
        context = AgentContext(
            timestamp=ts.to_pydatetime(),
            symbols=symbols,
            portfolio=portfolio_state,
            market_state=self.feed.market_snapshot(
                ts,
                symbols,
                lookback=self.config.agent.reasoning_lookback_bars,
                include_current=market_include_current,
            ),
            news_state=self.feed.news_snapshot(
                ts,
                symbols,
                lookback_hours=self.config.agent.news_lookback_hours,
                mode=self.config.data.news_mode,
            ),
            macro_state=self.feed.macro_snapshot(ts),
            fundamental_state=self.feed.fundamental_snapshot_for_symbols(symbols),
        )

        decisions = self.agent.decide(context)
        accepted_orders = 0
        new_positions = 0

        for decision in decisions:
            if accepted_orders >= self.config.backtest.max_orders_per_step:
                break
            if decision.confidence < self.config.agent.min_confidence:
                self._log_decision(ts, decision, False, "confidence_below_threshold")
                continue

            pre_qty = self._position_qty(decision.symbol.upper())
            order = self._decision_to_order(ts, decision, reference_prices, equity_ref)
            if order is None:
                self._log_decision(ts, decision, False, "risk_or_sizing_rejected")
                continue

            if order.side == "BUY" and pre_qty == 0:
                if new_positions >= self.config.agent.max_new_positions_per_step:
                    self._log_decision(ts, decision, False, "max_new_positions_reached")
                    continue
                new_positions += 1

            self.broker.queue_order(order)
            accepted_orders += 1
            self._log_decision(ts, decision, True, queue_reason)

    def run(self) -> BacktestResult:
        timeline = self.feed.timeline
        if not timeline:
            raise ValueError("No timeline data available. Download market data first.")

        last_decision_day = None
        decision_sessions = 0
        progress = tqdm(
            enumerate(timeline),
            total=len(timeline),
            desc="Backtest replay",
            unit="bar",
        )

        for step_idx, ts in progress:
            open_prices = self.feed.open_prices(ts)
            close_prices = self.feed.close_prices(ts)
            local_day = ts.tz_convert(self.config.data.timezone).date()

            should_rebalance = step_idx % self.config.backtest.rebalance_every_bars == 0
            if self.config.backtest.one_decision_per_day and local_day == last_decision_day:
                should_rebalance = False

            if self.config.backtest.decision_timing == "daily_open":
                if should_rebalance:
                    decision_sessions += 1
                    self._run_decision_session(
                        ts,
                        reference_prices=open_prices,
                        market_include_current=False,
                        queue_reason="queued_for_open",
                    )
                    last_decision_day = local_day
                self.broker.execute_pending(ts.to_pydatetime(), open_prices)
                self.broker.mark_to_market(ts.to_pydatetime(), close_prices)
                if step_idx % 25 == 0 and self.broker.equity_curve:
                    progress.set_postfix(
                        {
                            "equity": f"{self.broker.equity_curve[-1]['equity']:.0f}",
                            "fills": len(self.broker.fills),
                            "queued": len(self.broker.pending_orders),
                            "sessions": decision_sessions,
                        }
                    )
                continue

            self.broker.execute_pending(ts.to_pydatetime(), open_prices)
            self.broker.mark_to_market(ts.to_pydatetime(), close_prices)
            if should_rebalance:
                decision_sessions += 1
                self._run_decision_session(
                    ts,
                    reference_prices=close_prices,
                    market_include_current=True,
                    queue_reason="queued",
                )
                last_decision_day = local_day
            if step_idx % 25 == 0 and self.broker.equity_curve:
                progress.set_postfix(
                    {
                        "equity": f"{self.broker.equity_curve[-1]['equity']:.0f}",
                        "fills": len(self.broker.fills),
                        "queued": len(self.broker.pending_orders),
                        "sessions": decision_sessions,
                    }
                )

        progress.close()

        equity_curve = pd.DataFrame(self.broker.equity_curve)
        fills = pd.DataFrame(self.broker.fills_as_records())
        decisions = pd.DataFrame(self.decision_log)
        metrics = compute_backtest_metrics(equity_curve, fills)
        summary = {
            "start": str(timeline[0]),
            "end": str(timeline[-1]),
            "bars": len(timeline),
            "symbols": self.feed.available_symbols(),
            "final_equity": float(equity_curve["equity"].iloc[-1])
            if not equity_curve.empty
            else float(self.broker.cash),
            "cash": float(self.broker.cash),
            **metrics,
        }
        return BacktestResult(
            summary=summary,
            equity_curve=equity_curve,
            fills=fills,
            decisions=decisions,
        )

    @staticmethod
    def save(result: BacktestResult, output_dir: Path, run_name: str) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in run_name)
        summary_path = output_dir / f"{safe_name}_summary.json"
        equity_path = output_dir / f"{safe_name}_equity.csv"
        fills_path = output_dir / f"{safe_name}_fills.csv"
        decisions_path = output_dir / f"{safe_name}_decisions.csv"

        summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
        result.equity_curve.to_csv(equity_path, index=False)
        result.fills.to_csv(fills_path, index=False)
        result.decisions.to_csv(decisions_path, index=False)

        return {
            "summary": summary_path,
            "equity": equity_path,
            "fills": fills_path,
            "decisions": decisions_path,
        }
