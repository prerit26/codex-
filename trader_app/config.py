from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import List


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _parse_int(raw: str | None, default: int) -> int:
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _parse_list(raw: str | None, default: List[str]) -> List[str]:
    if raw is None or raw.strip() == "":
        return default
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


INDIA_LIQUID_UNIVERSE = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
    "ITC.NS",
    "BHARTIARTL.NS",
    "LT.NS",
    "AXISBANK.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "RELIANCE.BO",
    "TCS.BO",
    "HDFCBANK.BO",
    "ICICIBANK.BO",
    "INFY.BO",
    "SBIN.BO",
    "ITC.BO",
    "LT.BO",
    "AXISBANK.BO",
]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class APIConfig:
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    request_timeout_sec: int = 45
    temperature: float = 0.1
    max_tool_rounds: int = 6


@dataclass
class DataConfig:
    data_dir: Path = field(default_factory=lambda: _project_root() / "data")
    symbols: List[str] = field(default_factory=lambda: INDIA_LIQUID_UNIVERSE.copy())
    benchmark_symbol: str = "^NSEI"
    start_date: str = "2005-01-01"
    end_date: str = "2025-12-31"
    bar_interval: str = "1d"
    timezone: str = "Asia/Kolkata"
    news_enabled: bool = True
    news_provider: str = "google_rss"
    gdelt_max_records_per_day: int = 250
    gdelt_window_days: int = 30
    news_mode: str = "day_start_full"
    macro_enabled: bool = True
    macro_symbols: List[str] = field(
        default_factory=lambda: ["^NSEI", "^BSESN", "^INDIAVIX", "INRUSD=X", "CL=F", "GC=F", "^TNX"]
    )
    fundamentals_enabled: bool = True

    @property
    def market_dir(self) -> Path:
        return self.data_dir / "market"

    @property
    def news_dir(self) -> Path:
        return self.data_dir / "news"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def macro_dir(self) -> Path:
        return self.data_dir / "macro"

    @property
    def fundamentals_dir(self) -> Path:
        return self.data_dir / "fundamentals"


@dataclass
class AgentConfig:
    min_confidence: float = 0.45
    default_position_pct: float = 0.16
    max_position_pct: float = 0.40
    min_cash_reserve_pct: float = 0.0
    max_new_positions_per_step: int = 8
    reasoning_lookback_bars: int = 150
    news_lookback_hours: int = 24
    max_symbols_per_step: int = 24
    model_horizon_bars: int = 5
    model_min_samples: int = 600
    model_retrain_every_bars: int = 5
    model_ridge_alpha: float = 8.0
    model_max_training_samples: int = 12000
    model_entry_threshold: float = 0.0015
    model_exit_threshold: float = -0.0010
    model_vol_target: float = 0.16


@dataclass
class BacktestConfig:
    starting_cash: float = 100_000.0
    commission_bps: float = 2.0
    slippage_bps: float = 5.0
    max_gross_leverage: float = 3.0
    max_orders_per_step: int = 20
    rebalance_every_bars: int = 1
    decision_timing: str = "daily_open"
    one_decision_per_day: bool = True


@dataclass
class AppConfig:
    api: APIConfig
    data: DataConfig
    agent: AgentConfig
    backtest: BacktestConfig

    def ensure_dirs(self) -> None:
        self.data.data_dir.mkdir(parents=True, exist_ok=True)
        self.data.market_dir.mkdir(parents=True, exist_ok=True)
        self.data.news_dir.mkdir(parents=True, exist_ok=True)
        self.data.macro_dir.mkdir(parents=True, exist_ok=True)
        self.data.fundamentals_dir.mkdir(parents=True, exist_ok=True)
        self.data.outputs_dir.mkdir(parents=True, exist_ok=True)


def load_config(env_path: str | Path | None = None) -> AppConfig:
    if env_path is None:
        env_path = _project_root() / ".env"
    load_env_file(Path(env_path))

    api = APIConfig(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        request_timeout_sec=_parse_int(os.getenv("REQUEST_TIMEOUT_SEC"), 45),
        temperature=_parse_float(os.getenv("DEEPSEEK_TEMPERATURE"), 0.1),
        max_tool_rounds=_parse_int(os.getenv("MAX_TOOL_ROUNDS"), 6),
    )

    data = DataConfig(
        data_dir=Path(os.getenv("DATA_DIR", str(_project_root() / "data"))),
        symbols=_parse_list(
            os.getenv("SYMBOLS"),
            INDIA_LIQUID_UNIVERSE.copy(),
        ),
        benchmark_symbol=os.getenv("BENCHMARK_SYMBOL", "^NSEI").upper(),
        start_date=os.getenv("START_DATE", "2005-01-01"),
        end_date=os.getenv("END_DATE", "2025-12-31"),
        bar_interval=os.getenv("BAR_INTERVAL", "1d"),
        timezone=os.getenv("TIMEZONE", "Asia/Kolkata"),
        news_enabled=_parse_bool(os.getenv("NEWS_ENABLED"), True),
        news_provider=os.getenv("NEWS_PROVIDER", "google_rss").lower(),
        gdelt_max_records_per_day=_parse_int(
            os.getenv("GDELT_MAX_RECORDS_PER_DAY"), 250
        ),
        gdelt_window_days=_parse_int(os.getenv("GDELT_WINDOW_DAYS"), 30),
        news_mode=os.getenv("NEWS_MODE", "day_start_full"),
        macro_enabled=_parse_bool(os.getenv("MACRO_ENABLED"), True),
        macro_symbols=_parse_list(
            os.getenv("MACRO_SYMBOLS"),
            ["^NSEI", "^BSESN", "^INDIAVIX", "INRUSD=X", "CL=F", "GC=F", "^TNX"],
        ),
        fundamentals_enabled=_parse_bool(os.getenv("FUNDAMENTALS_ENABLED"), True),
    )

    agent = AgentConfig(
        min_confidence=_parse_float(os.getenv("MIN_CONFIDENCE"), 0.45),
        default_position_pct=_parse_float(os.getenv("DEFAULT_POSITION_PCT"), 0.16),
        max_position_pct=_parse_float(os.getenv("MAX_POSITION_PCT"), 0.40),
        min_cash_reserve_pct=_parse_float(os.getenv("MIN_CASH_RESERVE_PCT"), 0.0),
        max_new_positions_per_step=_parse_int(
            os.getenv("MAX_NEW_POSITIONS_PER_STEP"), 8
        ),
        reasoning_lookback_bars=_parse_int(
            os.getenv("REASONING_LOOKBACK_BARS"), 150
        ),
        news_lookback_hours=_parse_int(os.getenv("NEWS_LOOKBACK_HOURS"), 24),
        max_symbols_per_step=_parse_int(os.getenv("MAX_SYMBOLS_PER_STEP"), 24),
        model_horizon_bars=_parse_int(os.getenv("MODEL_HORIZON_BARS"), 5),
        model_min_samples=_parse_int(os.getenv("MODEL_MIN_SAMPLES"), 600),
        model_retrain_every_bars=_parse_int(
            os.getenv("MODEL_RETRAIN_EVERY_BARS"), 5
        ),
        model_ridge_alpha=_parse_float(os.getenv("MODEL_RIDGE_ALPHA"), 8.0),
        model_max_training_samples=_parse_int(
            os.getenv("MODEL_MAX_TRAINING_SAMPLES"), 12000
        ),
        model_entry_threshold=_parse_float(
            os.getenv("MODEL_ENTRY_THRESHOLD"), 0.0015
        ),
        model_exit_threshold=_parse_float(
            os.getenv("MODEL_EXIT_THRESHOLD"), -0.0010
        ),
        model_vol_target=_parse_float(os.getenv("MODEL_VOL_TARGET"), 0.16),
    )

    backtest = BacktestConfig(
        starting_cash=_parse_float(os.getenv("STARTING_CASH"), 100_000.0),
        commission_bps=_parse_float(os.getenv("COMMISSION_BPS"), 2.0),
        slippage_bps=_parse_float(os.getenv("SLIPPAGE_BPS"), 5.0),
        max_gross_leverage=_parse_float(os.getenv("MAX_GROSS_LEVERAGE"), 3.0),
        max_orders_per_step=_parse_int(os.getenv("MAX_ORDERS_PER_STEP"), 20),
        rebalance_every_bars=_parse_int(os.getenv("REBALANCE_EVERY_BARS"), 1),
        decision_timing=os.getenv("DECISION_TIMING", "daily_open").lower(),
        one_decision_per_day=_parse_bool(os.getenv("ONE_DECISION_PER_DAY"), True),
    )

    return AppConfig(api=api, data=data, agent=agent, backtest=backtest)
