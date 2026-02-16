from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _annualization_factor(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    deltas = index.to_series().diff().dropna().dt.total_seconds() / 86400.0
    median_days = float(deltas.median()) if not deltas.empty else 1.0
    if median_days <= 0:
        return 252.0
    return 365.25 / median_days


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    rolling_peak = equity.cummax()
    drawdowns = equity / rolling_peak - 1.0
    return float(drawdowns.min())


def _trade_stats(fills: pd.DataFrame) -> Dict[str, float]:
    if fills.empty:
        return {"closed_trades": 0, "win_rate": 0.0, "profit_factor": 0.0}

    lots: Dict[str, Dict[str, float]] = {}
    pnl_events: List[float] = []

    for _, row in fills.sort_values("timestamp").iterrows():
        symbol = row["symbol"]
        side = str(row["side"]).upper()
        qty = int(row["quantity"])
        price = float(row["price"])
        commission = float(row.get("commission", 0.0))
        lot = lots.get(symbol, {"qty": 0, "avg_price": 0.0})

        if side == "BUY":
            new_qty = lot["qty"] + qty
            if new_qty > 0:
                lot["avg_price"] = (
                    lot["avg_price"] * lot["qty"] + price * qty + commission
                ) / new_qty
            lot["qty"] = new_qty
            lots[symbol] = lot
        elif side == "SELL":
            sell_qty = min(qty, lot["qty"])
            if sell_qty <= 0:
                continue
            pnl = (price - lot["avg_price"]) * sell_qty - commission
            pnl_events.append(pnl)
            lot["qty"] -= sell_qty
            if lot["qty"] <= 0:
                lots.pop(symbol, None)
            else:
                lots[symbol] = lot

    if not pnl_events:
        return {"closed_trades": 0, "win_rate": 0.0, "profit_factor": 0.0}

    pnl_arr = np.array(pnl_events)
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = abs(float(losses.sum())) if losses.size else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    win_rate = float((pnl_arr > 0).mean())
    return {
        "closed_trades": int(len(pnl_events)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


def compute_backtest_metrics(equity_curve: pd.DataFrame, fills: pd.DataFrame) -> Dict[str, float]:
    if equity_curve.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "closed_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    curve = equity_curve.copy()
    curve["timestamp"] = pd.to_datetime(curve["timestamp"], utc=True)
    curve = curve.sort_values("timestamp").set_index("timestamp")
    returns = curve["equity"].pct_change().dropna()
    total_return = curve["equity"].iloc[-1] / curve["equity"].iloc[0] - 1.0

    ann_factor = _annualization_factor(curve.index)
    ann_return = (1.0 + total_return) ** (ann_factor / max(len(curve), 1)) - 1.0
    ann_vol = float(returns.std() * np.sqrt(ann_factor)) if not returns.empty else 0.0
    sharpe = float(returns.mean() / returns.std() * np.sqrt(ann_factor)) if returns.std() > 0 else 0.0
    max_dd = _max_drawdown(curve["equity"])

    trade_metrics = _trade_stats(fills)
    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "annualized_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "closed_trades": int(trade_metrics["closed_trades"]),
        "win_rate": float(trade_metrics["win_rate"]),
        "profit_factor": float(trade_metrics["profit_factor"]),
    }
