from __future__ import annotations

import argparse
from datetime import datetime

from trader_app.agents import DeepSeekTradingAgent, ModelAgent, RuleBasedAgent
from trader_app.backtest import BacktestEngine, ReplayDataFeed, SimBroker
from trader_app.config import load_config
from trader_app.data import (
    download_all_inputs,
    load_fundamental_snapshot,
    load_macro_data,
    load_market_data,
    load_news_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download data and run a historical replay backtest."
    )
    parser.add_argument("--agent", choices=["deepseek", "rule", "model"], default="deepseek")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    config.ensure_dirs()

    if not args.skip_download:
        report = download_all_inputs(config)
        print(
            f"Downloaded {len(report.market_files)} market files and {len(report.news_files)} news files"
        )
        if report.errors:
            print(f"Warnings: {len(report.errors)}")
            for line in report.errors[:15]:
                print(f"- {line}")

    market_data = load_market_data(config.data.market_dir, config.data.symbols)
    if not market_data:
        raise ValueError("No market data available after download.")
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
    engine = BacktestEngine(config, feed, broker, agent)
    result = engine.run()

    run_name = args.run_name or f"pipeline_{args.agent}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    paths = BacktestEngine.save(result, config.data.outputs_dir, run_name)
    print("Pipeline complete")
    print(f"Decision timing: {config.backtest.decision_timing}")
    print(f"One decision/day: {config.backtest.one_decision_per_day}")
    print(f"News mode: {config.data.news_mode}")
    for key, value in paths.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
