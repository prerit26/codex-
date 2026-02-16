from __future__ import annotations

import json
import re
from typing import Dict, List

import requests

from trader_app.config import AppConfig
from trader_app.types import AgentContext, TradeDecision
from .rule_based import RuleBasedAgent


class AgentToolbox:
    def __init__(self, context: AgentContext, config: AppConfig) -> None:
        self.context = context
        self.config = config

    def tool_specs(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_symbols",
                    "description": "Return symbols currently available in this step.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_portfolio_state",
                    "description": "Return portfolio cash, equity, open positions and pending orders.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_symbol_market_data",
                    "description": "Return market snapshot and recent bars for one symbol.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "bars": {"type": "integer", "minimum": 5, "maximum": 80},
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_symbol_news",
                    "description": "Return recent news rows for one symbol from historical news feed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_risk_constraints",
                    "description": "Return hard constraints for position sizing and trading limits.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_macro_state",
                    "description": "Return macro and cross-asset snapshot at current step.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_symbol_fundamentals",
                    "description": "Return latest fundamental snapshot fields for one symbol.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                        },
                        "required": ["symbol"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def execute(self, name: str, args: Dict) -> Dict:
        try:
            if name == "list_symbols":
                return {"symbols": self.context.symbols}
            if name == "get_portfolio_state":
                return self.context.portfolio
            if name == "get_symbol_market_data":
                symbol = str(args.get("symbol", "")).upper()
                bars = int(args.get("bars", 20))
                snapshot = self.context.market_state.get(symbol, {})
                if not snapshot:
                    return {"error": f"No market snapshot for {symbol}"}
                bars_payload = snapshot.get("bars", [])
                return {
                    "symbol": symbol,
                    "last_close": snapshot.get("last_close"),
                    "return_1d": snapshot.get("return_1d"),
                    "return_5d": snapshot.get("return_5d"),
                    "return_20d": snapshot.get("return_20d"),
                    "return_60d": snapshot.get("return_60d"),
                    "return_120d": snapshot.get("return_120d"),
                    "sma_20": snapshot.get("sma_20"),
                    "sma_50": snapshot.get("sma_50"),
                    "sma_100": snapshot.get("sma_100"),
                    "volatility_20d": snapshot.get("volatility_20d"),
                    "distance_from_60d_high": snapshot.get("distance_from_60d_high"),
                    "bars": bars_payload[-max(5, min(80, bars)) :],
                }
            if name == "get_symbol_news":
                symbol = str(args.get("symbol", "")).upper()
                limit = int(args.get("limit", 10))
                items = self.context.news_state.get(symbol, [])
                return {
                    "symbol": symbol,
                    "count": len(items),
                    "items": items[-max(1, min(50, limit)) :],
                }
            if name == "get_risk_constraints":
                return {
                    "min_confidence": self.config.agent.min_confidence,
                    "default_position_pct": self.config.agent.default_position_pct,
                    "max_position_pct": self.config.agent.max_position_pct,
                    "min_cash_reserve_pct": self.config.agent.min_cash_reserve_pct,
                    "max_new_positions_per_step": self.config.agent.max_new_positions_per_step,
                    "max_orders_per_step": self.config.backtest.max_orders_per_step,
                    "max_gross_leverage": self.config.backtest.max_gross_leverage,
                }
            if name == "get_macro_state":
                return self.context.macro_state
            if name == "get_symbol_fundamentals":
                symbol = str(args.get("symbol", "")).upper()
                return self.context.fundamental_state.get(symbol, {})
        except Exception as exc:
            return {"error": f"Tool {name} failed: {exc}"}
        return {"error": f"Unknown tool: {name}"}


class DeepSeekTradingAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.fallback = RuleBasedAgent(config)

    @property
    def _endpoint(self) -> str:
        base = self.config.api.deepseek_base_url.rstrip("/")
        return f"{base}/chat/completions"

    def _chat(self, messages: List[Dict], tools: List[Dict] | None = None) -> Dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api.deepseek_api_key}",
        }
        payload: Dict = {
            "model": self.config.api.deepseek_model,
            "messages": messages,
            "temperature": self.config.api.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = requests.post(
            self._endpoint,
            headers=headers,
            json=payload,
            timeout=self.config.api.request_timeout_sec,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"DeepSeek API error {response.status_code}: {response.text[:250]}"
            )
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("DeepSeek returned no choices")
        return choices[0].get("message", {})

    @staticmethod
    def _extract_json(content: str) -> Dict | None:
        try:
            return json.loads(content)
        except Exception:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _parse_decisions(self, content: str, allowed_symbols: List[str]) -> List[TradeDecision]:
        payload = self._extract_json(content)
        if not payload:
            return []
        raw_items = payload.get("decisions")
        if not isinstance(raw_items, list):
            return []

        allowed = {sym.upper() for sym in allowed_symbols}
        decisions: List[TradeDecision] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).upper()
            if symbol not in allowed:
                continue
            action = str(item.get("action", "HOLD")).upper()
            if action not in {"BUY", "SELL", "HOLD"}:
                action = "HOLD"
            confidence = float(item.get("confidence", 0.5))
            confidence = self._clamp(confidence, 0.0, 1.0)
            size_pct = float(item.get("size_pct", 0.0 if action == "HOLD" else self.config.agent.default_position_pct))
            size_pct = self._clamp(size_pct, 0.0, self.config.agent.max_position_pct if action == "BUY" else 1.0)
            rationale = str(item.get("rationale", ""))[:300]
            decisions.append(
                TradeDecision(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    size_pct=size_pct if action != "HOLD" else 0.0,
                    rationale=rationale,
                )
            )
        return decisions

    def _system_prompt(self) -> str:
        return (
            "You are a disciplined systematic trading decision engine for India equities (BSE/NSE). "
            "Goal: maximize return while respecting supplied constraints. "
            "A decision session happens once per day at market open. "
            "Assume full same-day news feed is available at that open by design. "
            "Use only provided context and tool outputs. "
            "Prefer high-conviction, low-churn actions with strong trend and confirmation. "
            "Respect risk constraints from tools. "
            "Return JSON with schema: "
            "{\"decisions\":[{\"symbol\":\"RELIANCE.NS\",\"action\":\"BUY|SELL|HOLD\","
            "\"confidence\":0.0,\"size_pct\":0.0,\"rationale\":\"...\"}]}. "
            "confidence is 0-1. size_pct for BUY is target fraction of equity for the position. "
            "For SELL, size_pct is fraction of current position to exit (0-1). "
            "If unclear edge, return HOLD."
        )

    def _user_prompt(self, context: AgentContext) -> str:
        compact_market = {}
        for symbol, snapshot in context.market_state.items():
            compact_market[symbol] = {
                "last_close": snapshot.get("last_close"),
                "return_1d": snapshot.get("return_1d"),
                "return_5d": snapshot.get("return_5d"),
                "return_20d": snapshot.get("return_20d"),
                "return_60d": snapshot.get("return_60d"),
                "return_120d": snapshot.get("return_120d"),
                "sma_20": snapshot.get("sma_20"),
                "sma_50": snapshot.get("sma_50"),
                "sma_100": snapshot.get("sma_100"),
                "volatility_20d": snapshot.get("volatility_20d"),
                "distance_from_60d_high": snapshot.get("distance_from_60d_high"),
                "news_count": len(context.news_state.get(symbol, [])),
            }
        payload = {
            "timestamp": context.timestamp.isoformat(),
            "symbols": context.symbols,
            "portfolio": context.portfolio,
            "market_compact": compact_market,
            "macro_compact": context.macro_state,
            "fundamental_available_for": [
                symbol
                for symbol, row in context.fundamental_state.items()
                if isinstance(row, dict) and row
            ],
            "instruction": (
                "Use tools if you need deeper bars/news/constraints, then return decisions JSON."
            ),
        }
        return json.dumps(payload, ensure_ascii=True)

    def decide(self, context: AgentContext) -> List[TradeDecision]:
        if not self.config.api.deepseek_api_key:
            return self.fallback.decide(context)

        toolbox = AgentToolbox(context, self.config)
        messages: List[Dict] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._user_prompt(context)},
        ]
        tools = toolbox.tool_specs()

        for _ in range(self.config.api.max_tool_rounds):
            try:
                message = self._chat(messages, tools=tools)
            except Exception:
                return self.fallback.decide(context)

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content"),
                        "tool_calls": tool_calls,
                    }
                )
                for call in tool_calls:
                    function_data = call.get("function", {})
                    name = function_data.get("name", "")
                    arg_text = function_data.get("arguments", "{}")
                    try:
                        args = json.loads(arg_text) if arg_text else {}
                    except Exception:
                        args = {}
                    result = toolbox.execute(name, args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "name": name,
                            "content": json.dumps(result, ensure_ascii=True),
                        }
                    )
                continue

            content = message.get("content", "") or ""
            decisions = self._parse_decisions(content, context.symbols)
            if decisions:
                return decisions
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Return strictly valid JSON with top-level key 'decisions' only."
                    ),
                }
            )

        return self.fallback.decide(context)
