from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from trader_app.config import AppConfig
from trader_app.types import AgentContext, TradeDecision
from .rule_based import RuleBasedAgent


@dataclass
class PendingSample:
    symbol: str
    step_idx: int
    features: np.ndarray
    entry_price: float
    entry_benchmark_price: float


class ModelAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.fallback = RuleBasedAgent(config)
        self.feature_names = [
            "ret_1d",
            "ret_5d",
            "ret_20d",
            "ret_60d",
            "ret_120d",
            "sma20_gap",
            "sma50_gap",
            "sma100_gap",
            "dist_high_60d",
            "vol_20d",
            "volume_rel",
            "news_score",
            "macro_nsei_20d",
            "macro_vix_level",
            "macro_inr_20d",
            "macro_crude_20d",
            "fund_pe",
            "fund_roe",
            "fund_rev_growth",
            "fund_debt",
            "fund_margin",
        ]
        self.step_idx = 0
        self.pending_samples: List[PendingSample] = []
        self.train_x: List[np.ndarray] = []
        self.train_y: List[float] = []
        self.weights: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.last_retrain_step: int = -10**9
        self.symbol_prices: Dict[str, Dict[int, float]] = {}
        self.benchmark_prices: Dict[int, float] = {}

    def _macro_row(self, context: AgentContext, key: str) -> Dict:
        macro = context.macro_state or {}
        return macro.get(key, {}) if isinstance(macro, dict) else {}

    def _benchmark_price(self, context: AgentContext) -> float:
        benchmark_keys = [
            self.config.data.benchmark_symbol,
            self.config.data.benchmark_symbol.replace("^", "_"),
            "^NSEI",
            "_NSEI",
        ]
        for key in benchmark_keys:
            row = self._macro_row(context, key)
            price = row.get("last_close") if row else None
            if price is not None:
                return float(price)
        market_row = context.market_state.get(self.config.data.benchmark_symbol, {})
        price = market_row.get("last_close")
        if price is None:
            return 0.0
        return float(price)

    def _lookup_price(self, series: Dict[int, float], target_step: int) -> Optional[float]:
        if target_step in series:
            return float(series[target_step])
        prior_steps = [idx for idx in series.keys() if idx <= target_step]
        if not prior_steps:
            return None
        return float(series[max(prior_steps)])

    def _news_score(self, rows: List[Dict]) -> float:
        if not rows:
            return 0.0
        joined = " ".join(str(row.get("title", "")).lower() for row in rows)
        positive_terms = ("beat", "growth", "upgrade", "surge", "profit", "strong", "bullish")
        negative_terms = ("miss", "downgrade", "loss", "weak", "fraud", "decline", "bearish")
        pos_hits = sum(1 for token in positive_terms if token in joined)
        neg_hits = sum(1 for token in negative_terms if token in joined)
        score = (pos_hits - neg_hits) / max(len(rows), 1)
        return float(np.clip(score, -2.0, 2.0))

    def _safe_float(self, value: object, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _feature_vector(self, symbol: str, snapshot: Dict, context: AgentContext) -> np.ndarray:
        last_close = max(1e-9, self._safe_float(snapshot.get("last_close")))
        sma_20 = self._safe_float(snapshot.get("sma_20"), last_close)
        sma_50 = self._safe_float(snapshot.get("sma_50"), last_close)
        sma_100 = self._safe_float(snapshot.get("sma_100"), last_close)
        bars = snapshot.get("bars", []) if isinstance(snapshot.get("bars"), list) else []
        volume_rel = 1.0
        if bars:
            volumes = [self._safe_float(row.get("volume"), 0.0) for row in bars]
            if volumes:
                avg_vol = np.mean(volumes)
                cur_vol = volumes[-1]
                if avg_vol > 0:
                    volume_rel = cur_vol / avg_vol

        macro_nsei = self._macro_row(context, "_NSEI") or self._macro_row(context, "^NSEI")
        macro_vix = self._macro_row(context, "_INDIAVIX") or self._macro_row(context, "^INDIAVIX")
        macro_inr = self._macro_row(context, "INRUSD=X")
        macro_crude = self._macro_row(context, "CL=F")
        fundamentals = context.fundamental_state.get(symbol, {}) if context.fundamental_state else {}

        feature_values = [
            self._safe_float(snapshot.get("return_1d")),
            self._safe_float(snapshot.get("return_5d")),
            self._safe_float(snapshot.get("return_20d")),
            self._safe_float(snapshot.get("return_60d")),
            self._safe_float(snapshot.get("return_120d")),
            (last_close / max(1e-9, sma_20)) - 1.0,
            (last_close / max(1e-9, sma_50)) - 1.0,
            (last_close / max(1e-9, sma_100)) - 1.0,
            self._safe_float(snapshot.get("distance_from_60d_high")),
            self._safe_float(snapshot.get("volatility_20d")),
            float(np.clip(volume_rel, 0.0, 4.0)),
            self._news_score(context.news_state.get(symbol, [])),
            self._safe_float(macro_nsei.get("return_20d")),
            self._safe_float(macro_vix.get("last_close")) / 100.0,
            self._safe_float(macro_inr.get("return_20d")),
            self._safe_float(macro_crude.get("return_20d")),
            self._safe_float(fundamentals.get("trailingPE")) / 100.0,
            self._safe_float(fundamentals.get("returnOnEquity")),
            self._safe_float(fundamentals.get("revenueGrowth")),
            self._safe_float(fundamentals.get("debtToEquity")) / 10.0,
            self._safe_float(fundamentals.get("profitMargins")),
        ]
        arr = np.array(feature_values, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _update_price_maps(self, context: AgentContext) -> None:
        benchmark_price = self._benchmark_price(context)
        if benchmark_price > 0:
            self.benchmark_prices[self.step_idx] = benchmark_price
        for symbol, snapshot in context.market_state.items():
            price = self._safe_float(snapshot.get("last_close"))
            if price <= 0:
                continue
            symbol_map = self.symbol_prices.setdefault(symbol, {})
            symbol_map[self.step_idx] = price

    def _finalize_pending_labels(self) -> None:
        horizon = self.config.agent.model_horizon_bars
        if horizon <= 0:
            return

        survivors: List[PendingSample] = []
        for sample in self.pending_samples:
            target_step = sample.step_idx + horizon
            if self.step_idx < target_step:
                survivors.append(sample)
                continue

            symbol_map = self.symbol_prices.get(sample.symbol, {})
            exit_symbol_price = self._lookup_price(symbol_map, target_step)
            if exit_symbol_price is None or sample.entry_price <= 0:
                continue

            symbol_log_return = np.log(exit_symbol_price / sample.entry_price)
            benchmark_label = symbol_log_return
            if sample.entry_benchmark_price > 0:
                exit_benchmark_price = self._lookup_price(self.benchmark_prices, target_step)
                if exit_benchmark_price and exit_benchmark_price > 0:
                    bench_log_return = np.log(exit_benchmark_price / sample.entry_benchmark_price)
                    benchmark_label = symbol_log_return - bench_log_return

            self.train_x.append(sample.features)
            self.train_y.append(float(benchmark_label))

        self.pending_samples = survivors
        max_samples = max(200, self.config.agent.model_max_training_samples)
        if len(self.train_y) > max_samples:
            self.train_x = self.train_x[-max_samples:]
            self.train_y = self.train_y[-max_samples:]

    def _retrain_if_needed(self) -> None:
        min_samples = self.config.agent.model_min_samples
        if len(self.train_y) < min_samples:
            return
        if self.step_idx - self.last_retrain_step < self.config.agent.model_retrain_every_bars:
            return

        x = np.vstack(self.train_x)
        y = np.array(self.train_y, dtype=float)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
        x_norm = (x - mean) / std

        centered_y = y - y.mean()
        alpha = max(1e-6, float(self.config.agent.model_ridge_alpha))
        reg = np.eye(x_norm.shape[1], dtype=float) * alpha
        lhs = x_norm.T @ x_norm + reg
        rhs = x_norm.T @ centered_y
        try:
            weights = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(lhs) @ rhs

        self.weights = weights
        self.intercept = float(y.mean())
        self.scaler_mean = mean
        self.scaler_std = std
        self.last_retrain_step = self.step_idx

    def _predict_alpha(self, feature_vec: np.ndarray) -> float:
        if self.weights is None or self.scaler_mean is None or self.scaler_std is None:
            return 0.0
        norm_vec = (feature_vec - self.scaler_mean) / self.scaler_std
        return float(self.intercept + norm_vec @ self.weights)

    def _held_symbols(self, context: AgentContext) -> set[str]:
        held: set[str] = set()
        for position in context.portfolio.get("positions", []):
            symbol = str(position.get("symbol", "")).upper()
            if symbol:
                held.add(symbol)
        return held

    def _build_model_decisions(
        self,
        context: AgentContext,
        features_map: Dict[str, np.ndarray],
    ) -> List[TradeDecision]:
        held_symbols = self._held_symbols(context)
        predictions: List[Dict] = []
        for symbol, feature_vec in features_map.items():
            snapshot = context.market_state.get(symbol, {})
            volatility = max(0.01, self._safe_float(snapshot.get("volatility_20d"), 0.2))
            alpha = self._predict_alpha(feature_vec)
            risk_adjusted = alpha / (volatility * volatility + 1e-6)
            predictions.append(
                {
                    "symbol": symbol,
                    "alpha": alpha,
                    "volatility": volatility,
                    "score": risk_adjusted,
                }
            )

        if not predictions:
            return []

        predictions.sort(key=lambda item: item["score"], reverse=True)
        score_values = np.array([row["score"] for row in predictions], dtype=float)
        score_min = float(score_values.min())
        score_max = float(score_values.max())
        span = max(1e-9, score_max - score_min)

        entry_threshold = self.config.agent.model_entry_threshold
        exit_threshold = self.config.agent.model_exit_threshold
        max_new = self.config.agent.max_new_positions_per_step
        decisions: List[TradeDecision] = []

        for row in predictions[: max_new * 2]:
            symbol = row["symbol"]
            alpha = row["alpha"]
            if alpha <= entry_threshold:
                continue
            rel_rank = (row["score"] - score_min) / span
            confidence = float(np.clip(0.5 + 0.45 * rel_rank, 0.0, 1.0))
            vol_scale = float(
                np.clip(
                    self.config.agent.model_vol_target / max(0.05, row["volatility"]),
                    0.6,
                    1.6,
                )
            )
            size_pct = self.config.agent.default_position_pct * vol_scale * (0.85 + 0.45 * rel_rank)
            size_pct = float(np.clip(size_pct, 0.02, self.config.agent.max_position_pct))
            decisions.append(
                TradeDecision(
                    symbol=symbol,
                    action="BUY",
                    confidence=confidence,
                    size_pct=size_pct,
                    rationale=f"Model alpha={alpha:.4f}, score={row['score']:.4f}",
                )
            )

        for row in predictions:
            symbol = row["symbol"]
            if symbol not in held_symbols:
                continue
            alpha = row["alpha"]
            if alpha >= exit_threshold:
                continue
            confidence = float(np.clip(0.58 + min(0.35, abs(alpha) * 12.0), 0.0, 1.0))
            full_exit = alpha < (exit_threshold * 2.0)
            decisions.append(
                TradeDecision(
                    symbol=symbol,
                    action="SELL",
                    confidence=confidence,
                    size_pct=1.0 if full_exit else 0.5,
                    rationale=f"Model alpha={alpha:.4f} below exit threshold",
                )
            )

        return decisions

    def decide(self, context: AgentContext) -> List[TradeDecision]:
        if not context.market_state:
            return []

        self._update_price_maps(context)
        self._finalize_pending_labels()
        self._retrain_if_needed()

        features_map: Dict[str, np.ndarray] = {}
        for symbol, snapshot in context.market_state.items():
            features_map[symbol] = self._feature_vector(symbol, snapshot, context)

        if self.weights is None:
            decisions = self.fallback.decide(context)
        else:
            decisions = self._build_model_decisions(context, features_map)
            if not decisions:
                decisions = self.fallback.decide(context)

        benchmark_entry = self._benchmark_price(context)
        for symbol, feature_vec in features_map.items():
            entry_price = self._safe_float(context.market_state.get(symbol, {}).get("last_close"))
            if entry_price <= 0:
                continue
            self.pending_samples.append(
                PendingSample(
                    symbol=symbol,
                    step_idx=self.step_idx,
                    features=feature_vec,
                    entry_price=entry_price,
                    entry_benchmark_price=benchmark_entry,
                )
            )

        self.step_idx += 1
        return decisions
