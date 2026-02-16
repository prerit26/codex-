# Agentic Trader App (Historical Replay + DeepSeek)

This app simulates a live trading loop using historical market/news inputs.
Default mode is optimized for India (BSE/NSE) with daily bars from 2005-01-01 to 2025-12-31.

## What It Includes

- DeepSeek trading agent with tool-calling (`list_symbols`, `get_portfolio_state`, `get_symbol_market_data`, `get_symbol_news`, `get_risk_constraints`).
- Historical market downloader (Yahoo Finance).
- Historical news downloader with provider switch (`google_rss` default, `gdelt` optional).
- Macro/cross-asset downloader (indices, VIX, FX, commodities, rates).
- Fundamental snapshot downloader (valuation, quality, growth, leverage metrics).
- Event-driven replay engine:
  - One decision session per day at market open.
  - Day news can be injected at day start (`NEWS_MODE=day_start_full`).
  - Orders are submitted and filled at the same day open.
  - Portfolio is marked to close each day.
  - Leverage and sizing limits are enforced before queueing.
- Metrics and artifacts:
  - `summary.json`
  - `equity.csv`
  - `fills.csv`
  - `decisions.csv`

## Folder Structure

```text
trader_app/
  trader_app/
    agents/
    backtest/
    data/
    config.py
    types.py
  scripts/
    download_data.py
    run_backtest.py
    run_pipeline.py
  .env.example
  requirements.txt
```

## Setup

1. Install dependencies:

```powershell
cd C:\Users\preri\Desktop\COVERS\trader_app
py -m pip install -r requirements.txt
```

2. Create `.env` from template and set your DeepSeek key:

```powershell
Copy-Item .env.example .env
```

Set at least:

- `DEEPSEEK_API_KEY`
- `SYMBOLS`
- `START_DATE`
- `END_DATE`

## Download Historical Inputs

```powershell
py scripts\download_data.py
```

This writes:

- `data/market/<SYMBOL>.csv`
- `data/news/<SYMBOL>_news.csv` (if `NEWS_ENABLED=true`)
- `data/macro/*.csv` (if `MACRO_ENABLED=true`)
- `data/fundamentals/fundamental_snapshot.csv` (if `FUNDAMENTALS_ENABLED=true`)

## Run Replay Backtest

Rule-based baseline:

```powershell
py scripts\run_backtest.py --agent rule --run-name baseline_rule
```

Model-based agent (no LLM calls):

```powershell
py scripts\run_backtest.py --agent model --run-name model_v1
```

DeepSeek agent:

```powershell
py scripts\run_backtest.py --agent deepseek --run-name deepseek_v1
```

Single command pipeline (download + backtest):

```powershell
py scripts\run_pipeline.py --agent deepseek --run-name deepseek_pipeline
```

Results are written to:

- `data/outputs/<run>_summary.json`
- `data/outputs/<run>_equity.csv`
- `data/outputs/<run>_fills.csv`
- `data/outputs/<run>_decisions.csv`

## Key Configuration

In `.env`:

- `MIN_CONFIDENCE`
- `DEFAULT_POSITION_PCT`
- `MAX_POSITION_PCT`
- `MIN_CASH_RESERVE_PCT`
- `MAX_GROSS_LEVERAGE`
- `COMMISSION_BPS`
- `SLIPPAGE_BPS`
- `REBALANCE_EVERY_BARS`
- `DECISION_TIMING` (`daily_open` by default)
- `ONE_DECISION_PER_DAY` (`true` by default)
- `NEWS_MODE` (`day_start_full` by default)
- `GDELT_WINDOW_DAYS` (increase for faster long-range downloads)
- `NEWS_PROVIDER` (`google_rss` default, `gdelt` optional)
- `MACRO_ENABLED`, `MACRO_SYMBOLS`
- `FUNDAMENTALS_ENABLED`
- `MODEL_*` fields for training horizon/retrain/thresholds

## Notes

- This is for research/backtesting, not financial advice.
- News coverage from GDELT is broad and may include noise.
- Defaults are tuned for an aggressive India-market replay profile.
- DeepSeek failure or missing key automatically falls back to rule-based decisions.
