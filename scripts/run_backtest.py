from __future__ import annotations

import argparse
from datetime import datetime

from trader_app.agents import DeepSeekTradingAgent, ModelAgent, RuleBasedAgent
from trader_app.backtest import BacktestEngine, ReplayDataFeed, SimBroker
from trader_app.config import load_config
from trader_app.data import (
    load_fundamental_snapshot,
    load_macro_data,
    load_market_data,
    load_news_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run historical replay backtest using DeepSeek or rule-based agent."
    )
    parser.add_argument(
        "--agent",
        choices=["deepseek", "rule", "model"],
        default="deepseek",
        help="Agent type to run.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional label for output filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    config.ensure_dirs()

    market_data = load_market_data(config.data.market_dir, config.data.symbols)
    if not market_data:
        raise ValueError(
            "No market files found. Run `py scripts/download_data.py` first."
        )
    news_data = load_news_data(config.data.news_dir, config.data.symbols)
    macro_data = load_macro_data(config.data.macro_dir)
    fundamental_snapshot = load_fundamental_snapshot(config.data.fundamentals_dir)

    feed = ReplayDataFeed(
        market_data=market_data,
        news_data=news_data,
        macro_data=macro_data,
        fundamental_snapshot=fundamental_snapshot,
        benchmark_symbol=config.data.benchmark_symbol,
        timezone=config.data.timezone,
        news_mode=config.data.news_mode,
    )
    broker = SimBroker(
        starting_cash=config.backtest.starting_cash,
        commission_bps=config.backtest.commission_bps,
        slippage_bps=config.backtest.slippage_bps,
    )
    if args.agent == "rule":
        agent = RuleBasedAgent(config)
    elif args.agent == "model":
        agent = ModelAgent(config)
    else:
        agent = DeepSeekTradingAgent(config)

    engine = BacktestEngine(config=config, feed=feed, broker=broker, agent=agent)
    result = engine.run()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = (
        args.run_name
        if args.run_name
        else f"{args.agent}_{config.data.start_date}_{config.data.end_date}_{timestamp}"
    )
    output_paths = BacktestEngine.save(result, config.data.outputs_dir, run_name)

    print("\nBacktest complete")
    print(f"Agent: {args.agent}")
    print(f"Decision timing: {config.backtest.decision_timing}")
    print(f"One decision/day: {config.backtest.one_decision_per_day}")
    print(f"News mode: {config.data.news_mode}")
    for key, value in result.summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print("\nOutput files:")
    for label, path in output_paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
