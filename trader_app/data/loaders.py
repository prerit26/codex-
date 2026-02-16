from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def load_market_data(market_dir: Path, symbols: Iterable[str]) -> Dict[str, pd.DataFrame]:
    symbol_data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = market_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df.index = pd.to_datetime(df.index, utc=True)
        symbol_data[symbol] = df
    return symbol_data


def _normalize_news_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "seen_date" in normalized.columns:
        normalized["seen_date"] = pd.to_datetime(
            normalized["seen_date"], errors="coerce", utc=True
        )
    elif "published_at" in normalized.columns:
        normalized["seen_date"] = pd.to_datetime(
            normalized["published_at"], errors="coerce", utc=True
        )
    else:
        normalized["seen_date"] = pd.NaT
    return normalized.sort_values("seen_date")


def load_news_data(news_dir: Path, symbols: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for symbol in symbols:
        path = news_dir / f"{symbol}_news.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["symbol"] = symbol
        frames.append(_normalize_news_df(frame))
    if not frames:
        return pd.DataFrame(
            columns=["symbol", "seen_date", "title", "url", "source", "domain"]
        )
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("seen_date").reset_index(drop=True)


def load_macro_data(macro_dir: Path) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    if not macro_dir.exists():
        return data
    for path in macro_dir.glob("*.csv"):
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df = df.set_index("timestamp").sort_index()
            df.index = pd.to_datetime(df.index, utc=True)
            data[path.stem] = df
        except Exception:
            continue
    return data


def load_fundamental_snapshot(fundamentals_dir: Path) -> pd.DataFrame:
    path = fundamentals_dir / "fundamental_snapshot.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
