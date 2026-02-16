from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Dict, List

from trader_app.types import Fill, Order, Position


class SimBroker:
    def __init__(
        self,
        starting_cash: float,
        commission_bps: float,
        slippage_bps: float,
    ) -> None:
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Order] = []
        self.fills: List[Fill] = []
        self.realized_pnl = 0.0
        self.equity_curve: List[Dict] = []

    def queue_order(self, order: Order) -> None:
        if order.quantity <= 0:
            return
        side = order.side.upper()
        if side not in {"BUY", "SELL"}:
            return
        self.pending_orders.append(order)

    def _slipped_price(self, side: str, open_price: float) -> float:
        bps = self.slippage_bps / 10_000.0
        if side == "BUY":
            return open_price * (1.0 + bps)
        return open_price * (1.0 - bps)

    def _commission(self, notional: float) -> float:
        return abs(notional) * self.commission_bps / 10_000.0

    def execute_pending(self, timestamp: datetime, open_prices: Dict[str, float]) -> List[Fill]:
        if not self.pending_orders:
            return []

        executed: List[Fill] = []
        remaining: List[Order] = []

        for order in self.pending_orders:
            symbol = order.symbol
            if symbol not in open_prices:
                remaining.append(order)
                continue

            open_price = float(open_prices[symbol])
            fill_price = self._slipped_price(order.side, open_price)
            quantity = int(order.quantity)
            if quantity <= 0:
                continue

            position = self.positions.get(symbol, Position(symbol=symbol))
            side = order.side.upper()

            if side == "BUY":
                notional = fill_price * quantity
                commission = self._commission(notional)
                total_cost = notional + commission
                self.cash -= total_cost

                new_qty = position.quantity + quantity
                if new_qty > 0:
                    position.avg_price = (
                        (position.avg_price * position.quantity + fill_price * quantity)
                        / new_qty
                    )
                position.quantity = new_qty
                self.positions[symbol] = position

            else:
                sell_qty = min(quantity, position.quantity)
                if sell_qty <= 0:
                    continue
                notional = fill_price * sell_qty
                commission = self._commission(notional)
                proceeds = notional - commission
                self.cash += proceeds
                self.realized_pnl += (fill_price - position.avg_price) * sell_qty - commission
                position.quantity -= sell_qty
                if position.quantity <= 0:
                    self.positions.pop(symbol, None)
                else:
                    self.positions[symbol] = position
                quantity = sell_qty

            fill = Fill(
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=fill_price,
                notional=fill_price * quantity,
                commission=self._commission(fill_price * quantity),
                slippage_bps=self.slippage_bps,
                rationale=order.rationale,
            )
            self.fills.append(fill)
            executed.append(fill)

        self.pending_orders = remaining
        return executed

    def _position_value(self, symbol: str, last_price: float) -> float:
        position = self.positions.get(symbol)
        if position is None:
            return 0.0
        return position.quantity * last_price

    def mark_to_market(self, timestamp: datetime, close_prices: Dict[str, float]) -> float:
        market_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in close_prices:
                market_value += position.quantity * float(close_prices[symbol])
            else:
                market_value += position.quantity * position.avg_price
        equity = self.cash + market_value
        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "cash": self.cash,
                "market_value": market_value,
                "equity": equity,
            }
        )
        return equity

    def gross_exposure(self, close_prices: Dict[str, float]) -> float:
        exposure = 0.0
        for symbol, position in self.positions.items():
            price = close_prices.get(symbol, position.avg_price)
            exposure += abs(position.quantity * price)
        return exposure

    def portfolio_snapshot(self, close_prices: Dict[str, float]) -> Dict:
        positions_payload: List[Dict] = []
        for symbol, position in self.positions.items():
            mark = close_prices.get(symbol, position.avg_price)
            market_value = position.quantity * mark
            unrealized = (mark - position.avg_price) * position.quantity
            positions_payload.append(
                {
                    "symbol": symbol,
                    "quantity": position.quantity,
                    "avg_price": round(position.avg_price, 6),
                    "mark_price": round(mark, 6),
                    "market_value": round(market_value, 6),
                    "unrealized_pnl": round(unrealized, 6),
                }
            )

        equity = self.cash + sum(item["market_value"] for item in positions_payload)
        return {
            "cash": round(self.cash, 6),
            "equity": round(equity, 6),
            "realized_pnl": round(self.realized_pnl, 6),
            "positions": positions_payload,
            "pending_order_count": len(self.pending_orders),
        }

    def fills_as_records(self) -> List[Dict]:
        return [asdict(fill) for fill in self.fills]
