from __future__ import annotations

from trader_app.config import load_config
from trader_app.data import download_all_inputs


def main() -> None:
    config = load_config()
    report = download_all_inputs(config)

    print("\nData download complete")
    print(f"Market files: {len(report.market_files)}")
    print(f"News files: {len(report.news_files)}")
    print(f"Macro files: {len(report.macro_files)}")
    print(f"Fundamental files: {len(report.fundamental_files)}")
    if report.errors:
        print(f"Warnings/Errors: {len(report.errors)}")
        for line in report.errors[:20]:
            print(f"- {line}")


if __name__ == "__main__":
    main()
