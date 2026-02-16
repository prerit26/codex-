from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests
from tqdm import tqdm
import yfinance as yf

from trader_app.config import AppConfig


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"


SYMBOL_QUERY_OVERRIDES = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "INFY": "Infosys",
    "SBIN": "State Bank of India",
    "ITC": "ITC Limited",
    "BHARTIARTL": "Bharti Airtel",
    "LT": "Larsen and Toubro",
    "AXISBANK": "Axis Bank",
    "BAJFINANCE": "Bajaj Finance",
    "MARUTI": "Maruti Suzuki",
    "SUNPHARMA": "Sun Pharmaceutical",
    "TITAN": "Titan Company",
    "ULTRACEMCO": "UltraTech Cement",
    "^NSEI": "Nifty 50",
    "^BSESN": "BSE Sensex",
}


FUNDAMENTAL_FIELDS = [
    "marketCap",
    "enterpriseValue",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "beta",
    "profitMargins",
    "operatingMargins",
    "returnOnAssets",
    "returnOnEquity",
    "debtToEquity",
    "currentRatio",
    "quickRatio",
    "revenueGrowth",
    "earningsGrowth",
]


@dataclass
class DownloadReport:
    market_files: List[Path]
    news_files: List[Path]
    macro_files: List[Path]
    fundamental_files: List[Path]
    errors: List[str]


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized = normalized.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    normalized.index = pd.to_datetime(normalized.index, utc=True).tz_convert("UTC")
    normalized.index.name = "timestamp"
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in cols:
        if col not in normalized.columns:
            normalized[col] = pd.NA
    normalized = normalized[cols].dropna(subset=["open", "high", "low", "close"])
    return normalized.sort_index()


def _iter_windows(start: datetime, end: datetime, window_days: int) -> List[tuple[datetime, datetime]]:
    windows: List[tuple[datetime, datetime]] = []
    current = start
    while current <= end:
        window_end = min(current + timedelta(days=window_days - 1), end)
        windows.append((current, window_end))
        current = window_end + timedelta(days=1)
    return windows


def _to_gdelt_dt(dt: datetime, at_end_of_day: bool) -> str:
    if at_end_of_day:
        return dt.strftime("%Y%m%d235959")
    return dt.strftime("%Y%m%d000000")


def _symbol_query(symbol: str) -> str:
    upper_symbol = symbol.upper()
    base = upper_symbol.split(".")[0]
    if upper_symbol in SYMBOL_QUERY_OVERRIDES:
        name = SYMBOL_QUERY_OVERRIDES[upper_symbol]
    elif base in SYMBOL_QUERY_OVERRIDES:
        name = SYMBOL_QUERY_OVERRIDES[base]
    else:
        name = base
    return f"{name} India stock"


def download_market_history(config: AppConfig) -> tuple[List[Path], List[str]]:
    saved_files: List[Path] = []
    errors: List[str] = []
    config.data.market_dir.mkdir(parents=True, exist_ok=True)

    for symbol in tqdm(config.data.symbols, desc="Downloading market data"):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=config.data.start_date,
                end=config.data.end_date,
                interval=config.data.bar_interval,
                auto_adjust=False,
                actions=False,
            )
            if hist.empty:
                errors.append(f"No market data for {symbol}")
                continue
            normalized = _normalize_ohlcv(hist)
            out_path = config.data.market_dir / f"{symbol}.csv"
            normalized.to_csv(out_path)
            saved_files.append(out_path)
        except Exception as exc:
            errors.append(f"{symbol} market download failed: {exc}")

    return saved_files, errors


def download_macro_history(config: AppConfig) -> tuple[List[Path], List[str]]:
    if not config.data.macro_enabled:
        return [], []
    config.data.macro_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    errors: List[str] = []
    for symbol in tqdm(config.data.macro_symbols, desc="Downloading macro data"):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=config.data.start_date,
                end=config.data.end_date,
                interval=config.data.bar_interval,
                auto_adjust=False,
                actions=False,
            )
            if hist.empty:
                errors.append(f"No macro data for {symbol}")
                continue
            normalized = _normalize_ohlcv(hist)
            out_path = config.data.macro_dir / f"{symbol.replace('^', '_')}.csv"
            normalized.to_csv(out_path)
            files.append(out_path)
        except Exception as exc:
            errors.append(f"{symbol} macro download failed: {exc}")
    return files, errors


def _serialize_statement(df: pd.DataFrame, symbol: str, statement_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.transpose().reset_index().rename(columns={"index": "statement_date"})
    work["symbol"] = symbol
    work["statement_type"] = statement_name
    work["statement_date"] = pd.to_datetime(work["statement_date"], errors="coerce", utc=True)
    return work


def download_fundamentals(config: AppConfig) -> tuple[List[Path], List[str]]:
    if not config.data.fundamentals_enabled:
        return [], []
    config.data.fundamentals_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    errors: List[str] = []

    snapshot_rows: List[Dict] = []
    statement_rows: List[pd.DataFrame] = []

    for symbol in tqdm(config.data.symbols, desc="Downloading fundamentals"):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            snapshot = {"symbol": symbol, "snapshot_time": datetime.utcnow().isoformat()}
            for field in FUNDAMENTAL_FIELDS:
                snapshot[field] = info.get(field)
            snapshot_rows.append(snapshot)

            statement_rows.append(_serialize_statement(ticker.quarterly_income_stmt, symbol, "quarterly_income"))
            statement_rows.append(_serialize_statement(ticker.quarterly_balance_sheet, symbol, "quarterly_balance_sheet"))
            statement_rows.append(_serialize_statement(ticker.quarterly_cashflow, symbol, "quarterly_cashflow"))
        except Exception as exc:
            errors.append(f"{symbol} fundamentals download failed: {exc}")

    snapshot_path = config.data.fundamentals_dir / "fundamental_snapshot.csv"
    pd.DataFrame(snapshot_rows).to_csv(snapshot_path, index=False)
    files.append(snapshot_path)

    statement_df = pd.concat([x for x in statement_rows if not x.empty], ignore_index=True) if statement_rows else pd.DataFrame()
    statement_path = config.data.fundamentals_dir / "fundamental_statements.csv"
    statement_df.to_csv(statement_path, index=False)
    files.append(statement_path)
    return files, errors


def _fetch_gdelt_articles(
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    max_records: int,
    timeout_sec: int,
) -> List[Dict]:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "DateDesc",
        "maxrecords": str(max_records),
        "startdatetime": _to_gdelt_dt(start_dt, at_end_of_day=False),
        "enddatetime": _to_gdelt_dt(end_dt, at_end_of_day=True),
    }
    response = requests.get(GDELT_DOC_URL, params=params, timeout=timeout_sec)
    if response.status_code != 200:
        return []
    payload = response.json()
    return payload.get("articles", []) if isinstance(payload, dict) else []


def _fetch_google_rss_articles(
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    timeout_sec: int,
) -> List[Dict]:
    q = f'{query} after:{start_dt.date().isoformat()} before:{(end_dt + timedelta(days=1)).date().isoformat()}'
    url = (
        f"{GOOGLE_NEWS_RSS_URL}"
        f"?q={quote_plus(q)}&hl=en-IN&gl=IN&ceid=IN:en"
    )
    parsed = feedparser.parse(url)
    rows: List[Dict] = []
    for entry in parsed.entries:
        published = entry.get("published", "") or entry.get("updated", "")
        rows.append(
            {
                "seen_date": published,
                "source": entry.get("source", {}).get("title")
                if isinstance(entry.get("source"), dict)
                else None,
                "domain": None,
                "language": "en",
                "title": entry.get("title"),
                "url": entry.get("link"),
                "social_image": None,
            }
        )
    return rows


def download_news_history(config: AppConfig) -> tuple[List[Path], List[str]]:
    if not config.data.news_enabled:
        return [], []

    config.data.news_dir.mkdir(parents=True, exist_ok=True)
    start_dt = datetime.fromisoformat(config.data.start_date)
    end_dt = datetime.fromisoformat(config.data.end_date)
    windows = _iter_windows(
        start_dt,
        end_dt,
        window_days=max(1, int(config.data.gdelt_window_days)),
    )

    saved_files: List[Path] = []
    errors: List[str] = []
    provider = config.data.news_provider.lower()

    for idx, symbol in enumerate(tqdm(config.data.symbols, desc="Downloading news data")):
        print(f"[news] {symbol} ({idx + 1}/{len(config.data.symbols)}) via {provider}")
        rows: List[Dict] = []
        query = _symbol_query(symbol)
        for win_start, win_end in windows:
            try:
                if provider == "gdelt":
                    articles = _fetch_gdelt_articles(
                        query=f'"{query}" AND (stock OR market OR earnings)',
                        start_dt=win_start,
                        end_dt=win_end,
                        max_records=config.data.gdelt_max_records_per_day,
                        timeout_sec=config.api.request_timeout_sec,
                    )
                    for item in articles:
                        rows.append(
                            {
                                "symbol": symbol,
                                "query": query,
                                "seen_date": item.get("seendate"),
                                "source": item.get("sourcecountry"),
                                "domain": item.get("domain"),
                                "language": item.get("language"),
                                "title": item.get("title"),
                                "url": item.get("url"),
                                "social_image": item.get("socialimage"),
                            }
                        )
                else:
                    articles = _fetch_google_rss_articles(
                        query=query,
                        start_dt=win_start,
                        end_dt=win_end,
                        timeout_sec=config.api.request_timeout_sec,
                    )
                    for item in articles:
                        rows.append(
                            {
                                "symbol": symbol,
                                "query": query,
                                **item,
                            }
                        )
            except Exception as exc:
                errors.append(
                    f"{symbol} news window {win_start.date()}-{win_end.date()} failed: {exc}"
                )

        if not rows:
            errors.append(f"No news rows collected for {symbol}")
            continue

        df = pd.DataFrame(rows)
        if provider == "google_rss" and "seen_date" in df.columns:
            df["seen_date"] = pd.to_datetime(df["seen_date"], errors="coerce", utc=True)
        if "url" in df.columns:
            df = df.drop_duplicates(subset=["url"], keep="first")
        out_path = config.data.news_dir / f"{symbol}_news.csv"
        df.to_csv(out_path, index=False)
        saved_files.append(out_path)

    return saved_files, errors


def download_all_inputs(config: AppConfig) -> DownloadReport:
    config.ensure_dirs()
    market_files, market_errors = download_market_history(config)
    news_files, news_errors = download_news_history(config)
    macro_files, macro_errors = download_macro_history(config)
    fundamental_files, fundamental_errors = download_fundamentals(config)
    return DownloadReport(
        market_files=market_files,
        news_files=news_files,
        macro_files=macro_files,
        fundamental_files=fundamental_files,
        errors=[*market_errors, *news_errors, *macro_errors, *fundamental_errors],
    )
