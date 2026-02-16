from __future__ import annotations

from datetime import timedelta
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _coerce_timestamp(ts: pd.Timestamp | str) -> pd.Timestamp:
    value = pd.Timestamp(ts)
    if value.tzinfo is None:
        return value.tz_localize("UTC")
    return value.tz_convert("UTC")


class ReplayDataFeed:
    def __init__(
        self,
        market_data: Dict[str, pd.DataFrame],
        news_data: pd.DataFrame,
        macro_data: Dict[str, pd.DataFrame] | None = None,
        fundamental_snapshot: pd.DataFrame | None = None,
        benchmark_symbol: str = "SPY",
        timezone: str = "UTC",
        news_mode: str = "rolling_lookback",
    ) -> None:
        self.market_data = market_data
        self.news_data = news_data.copy()
        self.macro_data = macro_data or {}
        self.fundamental_snapshot = (
            fundamental_snapshot.copy() if fundamental_snapshot is not None else pd.DataFrame()
        )
        self.benchmark_symbol = benchmark_symbol
        self.timezone = timezone
        self.news_mode = news_mode.lower()

        if "seen_date" in self.news_data.columns:
            self.news_data["seen_date"] = pd.to_datetime(
                self.news_data["seen_date"], utc=True, errors="coerce"
            )
        self.news_data = self.news_data.sort_values("seen_date")
        if not self.fundamental_snapshot.empty and "symbol" in self.fundamental_snapshot.columns:
            self.fundamental_snapshot["symbol"] = (
                self.fundamental_snapshot["symbol"].astype(str).str.upper()
            )

        self.timeline = self._build_timeline()

    def _build_timeline(self) -> List[pd.Timestamp]:
        benchmark_df = self.market_data.get(self.benchmark_symbol)
        if benchmark_df is not None and not benchmark_df.empty:
            return sorted(pd.to_datetime(benchmark_df.index, utc=True).unique().tolist())
        all_timestamps: set[pd.Timestamp] = set()
        for df in self.market_data.values():
            if df is None or df.empty:
                continue
            all_timestamps.update(pd.to_datetime(df.index, utc=True).tolist())
        return sorted(all_timestamps)

    def available_symbols(self) -> List[str]:
        return sorted(self.market_data.keys())

    def bar_at(self, symbol: str, timestamp: pd.Timestamp) -> Optional[pd.Series]:
        df = self.market_data.get(symbol)
        if df is None or df.empty:
            return None
        ts = _coerce_timestamp(timestamp)
        if ts not in df.index:
            return None
        return df.loc[ts]

    def open_prices(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        prices: Dict[str, float] = {}
        ts = _coerce_timestamp(timestamp)
        for symbol, df in self.market_data.items():
            if ts in df.index:
                prices[symbol] = float(df.loc[ts]["open"])
        return prices

    def close_prices(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        prices: Dict[str, float] = {}
        ts = _coerce_timestamp(timestamp)
        for symbol, df in self.market_data.items():
            if ts in df.index:
                prices[symbol] = float(df.loc[ts]["close"])
        return prices

    def history(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        lookback: int,
        include_current: bool = True,
    ) -> pd.DataFrame:
        df = self.market_data.get(symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        ts = _coerce_timestamp(timestamp)
        if include_current:
            hist = df[df.index <= ts].tail(lookback).copy()
        else:
            hist = df[df.index < ts].tail(lookback).copy()
        return hist

    def market_snapshot(
        self,
        timestamp: pd.Timestamp,
        symbols: Iterable[str],
        lookback: int,
        include_current: bool = True,
    ) -> Dict[str, Dict]:
        state: Dict[str, Dict] = {}
        ts = _coerce_timestamp(timestamp)
        for symbol in symbols:
            hist = self.history(symbol, ts, lookback, include_current=include_current)
            if hist.empty:
                continue
            close = hist["close"]
            latest = float(close.iloc[-1])
            ret_1d = float(close.pct_change().iloc[-1]) if len(hist) >= 2 else 0.0
            ret_5d = float(close.pct_change(5).iloc[-1]) if len(hist) >= 6 else 0.0
            ret_20d = float(close.pct_change(20).iloc[-1]) if len(hist) >= 21 else 0.0
            ret_60d = float(close.pct_change(60).iloc[-1]) if len(hist) >= 61 else 0.0
            ret_120d = float(close.pct_change(120).iloc[-1]) if len(hist) >= 121 else 0.0
            sma_20 = float(close.tail(20).mean()) if len(hist) >= 20 else latest
            sma_50 = float(close.tail(50).mean()) if len(hist) >= 50 else latest
            sma_100 = float(close.tail(100).mean()) if len(hist) >= 100 else latest
            vol_20 = (
                float(close.pct_change().tail(20).std() * np.sqrt(252))
                if len(hist) >= 20
                else 0.0
            )
            rolling_high_60 = float(close.tail(60).max()) if len(hist) >= 60 else latest
            dist_from_high_60 = (
                (latest / rolling_high_60) - 1.0 if rolling_high_60 > 0 else 0.0
            )
            bars_tail = (
                hist[["open", "high", "low", "close", "volume"]]
                .tail(min(lookback, 30))
                .reset_index()
            )
            bars_tail["timestamp"] = bars_tail["timestamp"].dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            state[symbol] = {
                "last_close": round(latest, 6),
                "return_1d": round(ret_1d, 6),
                "return_5d": round(ret_5d, 6),
                "return_20d": round(ret_20d, 6),
                "return_60d": round(ret_60d, 6),
                "return_120d": round(ret_120d, 6),
                "sma_20": round(sma_20, 6),
                "sma_50": round(sma_50, 6),
                "sma_100": round(sma_100, 6),
                "volatility_20d": round(vol_20, 6),
                "distance_from_60d_high": round(dist_from_high_60, 6),
                "bars": bars_tail.to_dict(orient="records"),
            }
        return state

    def news_snapshot(
        self,
        timestamp: pd.Timestamp,
        symbols: Iterable[str],
        lookback_hours: int,
        mode: str | None = None,
    ) -> Dict[str, List[Dict]]:
        if self.news_data.empty:
            return {symbol: [] for symbol in symbols}

        ts = _coerce_timestamp(timestamp)
        active_mode = (mode or self.news_mode).lower()

        if active_mode == "day_start_full":
            local_ts = ts.tz_convert(self.timezone)
            local_day = local_ts.date()
            local_seen = self.news_data["seen_date"].dt.tz_convert(self.timezone).dt.date
            filtered = self.news_data[local_seen == local_day]
        else:
            lower = ts - timedelta(hours=lookback_hours)
            filtered = self.news_data[
                (self.news_data["seen_date"] <= ts) & (self.news_data["seen_date"] >= lower)
            ]

        state: Dict[str, List[Dict]] = {}
        for symbol in symbols:
            rows = filtered[filtered["symbol"] == symbol].tail(30)
            payload_rows: List[Dict] = []
            for _, row in rows.iterrows():
                payload_rows.append(
                    {
                        "seen_date": row.get("seen_date").strftime("%Y-%m-%dT%H:%M:%SZ")
                        if pd.notna(row.get("seen_date"))
                        else None,
                        "title": row.get("title"),
                        "source": row.get("source"),
                        "domain": row.get("domain"),
                        "url": row.get("url"),
                    }
                )
            state[symbol] = payload_rows
        return state

    def macro_snapshot(self, timestamp: pd.Timestamp) -> Dict[str, Dict]:
        ts = _coerce_timestamp(timestamp)
        payload: Dict[str, Dict] = {}
        for macro_symbol, df in self.macro_data.items():
            hist = df[df.index <= ts].tail(120)
            if hist.empty:
                continue
            close = hist["close"]
            last_close = float(close.iloc[-1])
            ret_1d = float(close.pct_change().iloc[-1]) if len(close) >= 2 else 0.0
            ret_20d = float(close.pct_change(20).iloc[-1]) if len(close) >= 21 else 0.0
            vol_20d = (
                float(close.pct_change().tail(20).std() * np.sqrt(252))
                if len(close) >= 20
                else 0.0
            )
            payload[macro_symbol] = {
                "last_close": round(last_close, 6),
                "return_1d": round(ret_1d, 6),
                "return_20d": round(ret_20d, 6),
                "volatility_20d": round(vol_20d, 6),
            }
        return payload

    def fundamental_snapshot_for_symbols(self, symbols: Iterable[str]) -> Dict[str, Dict]:
        if self.fundamental_snapshot.empty:
            return {sym: {} for sym in symbols}
        payload: Dict[str, Dict] = {}
        for symbol in symbols:
            rows = self.fundamental_snapshot[
                self.fundamental_snapshot["symbol"] == symbol.upper()
            ]
            if rows.empty:
                payload[symbol] = {}
                continue
            row = rows.iloc[0].to_dict()
            payload[symbol] = row
        return payload
