"""
Hourly 2025 price histories for the revised pilot sample.

Run from the notebook after select_revised_sample has produced revised_df:

    from kalshi_price_history import collect_price_histories, plot_contracts

    prices_df, price_summary_df = collect_price_histories(revised_df)
    display(price_summary_df)
    plot_contracts(prices_df, revised_df, n=4)
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
STUDY_START = pd.Timestamp("2025-01-01", tz="UTC")
STUDY_END = pd.Timestamp("2026-01-01", tz="UTC")


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

def _get(url, params, max_retries=6):
    """GET with exponential backoff on rate limiting."""
    for attempt in range(max_retries):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 429:
            return response
        time.sleep(2 ** attempt)
    return response


def _endpoints(row):
    """Candlestick URLs to try, starting with the contract's own source."""
    historical = f"{BASE_URL}/historical/markets/{row['ticker']}/candlesticks"
    live = (
        f"{BASE_URL}/series/{row['series_ticker']}"
        f"/markets/{row['ticker']}/candlesticks"
    )
    if row.get("data_source") == "live":
        return [live, historical]
    return [historical, live]


def _field(raw, names):
    """Return the first column that exists; the live and historical
    endpoints name some candlestick fields differently."""
    for name in names:
        if name in raw.columns:
            return raw[name]
    return pd.Series(np.nan, index=raw.index, dtype="object")


def _close_value(field):
    """Closing price in dollars from a nested candlestick field."""
    if not isinstance(field, dict):
        return np.nan
    value = pd.to_numeric(field.get("close_dollars", field.get("close")),
                          errors="coerce")
    if pd.notna(value) and value > 1:   # some responses are in cents
        value = value / 100
    return value


def fetch_candlesticks(row, period_interval=60, chunk_hours=3000):
    """Fetch hourly candlesticks for the part of a contract's life in 2025."""
    start = max(pd.Timestamp(row["created_time"]), STUDY_START)
    end = min(pd.Timestamp(row["effective_end_time"]), STUDY_END)
    start_ts, end_ts = int(start.timestamp()), int(end.timestamp())
    step = chunk_hours * 3600

    for url in _endpoints(row):
        candles, failed = [], False
        for chunk_start in range(start_ts, end_ts, step):
            response = _get(url, {
                "start_ts": chunk_start,
                "end_ts": min(chunk_start + step, end_ts),
                "period_interval": period_interval,
            })
            if response.status_code != 200:
                failed = True
                break
            candles.extend(response.json().get("candlesticks", []))
            time.sleep(0.2)
        if not failed:
            break
    else:
        print(f"  {row['ticker']}: no candlesticks from either endpoint "
              f"(last status {response.status_code})")
        return pd.DataFrame()

    if not candles:
        return pd.DataFrame()

    raw = pd.DataFrame(candles)
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(raw["end_period_ts"], unit="s", utc=True),
        "yes_bid": _field(raw, ["yes_bid"]).apply(_close_value),
        "yes_ask": _field(raw, ["yes_ask"]).apply(_close_value),
        "trade_price": _field(raw, ["price"]).apply(_close_value),
        "volume": pd.to_numeric(
            _field(raw, ["volume", "volume_fp"]), errors="coerce"),
        "open_interest": pd.to_numeric(
            _field(raw, ["open_interest", "open_interest_fp"]), errors="coerce"),
    })
    out["mid_price"] = out[["yes_bid", "yes_ask"]].mean(axis=1)
    out["spread"] = out["yes_ask"] - out["yes_bid"]
    return out.drop_duplicates("timestamp").sort_values("timestamp")


def to_hourly_grid(candles):
    """Fill hours with no activity: prices carry forward, volume is zero."""
    if candles.empty:
        return candles
    candles = candles.set_index("timestamp")
    grid = pd.date_range(candles.index.min(), candles.index.max(), freq="h")
    candles = candles.reindex(grid)
    candles["had_trade"] = candles["volume"].fillna(0) > 0
    candles["volume"] = candles["volume"].fillna(0)
    carry = ["yes_bid", "yes_ask", "trade_price", "mid_price",
             "spread", "open_interest"]
    candles[carry] = candles[carry].ffill()
    return candles.rename_axis("timestamp").reset_index()


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_price_histories(contracts_df, out_dir="data/pilot"):
    """Fetch, regularise and save 2025 hourly prices for every contract."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contracts = contracts_df.copy()
    if "series_ticker" not in contracts:
        contracts["series_ticker"] = contracts["event_ticker"].str.split("-").str[0]
    for col in ["created_time", "effective_end_time"]:
        contracts[col] = pd.to_datetime(contracts[col], utc=True)

    frames, summary = [], []
    for i, row in enumerate(contracts.to_dict("records"), start=1):
        print(f"{i}/{len(contracts)}: {row['ticker']}")
        prices = to_hourly_grid(fetch_candlesticks(row))
        if prices.empty:
            continue
        prices.insert(0, "ticker", row["ticker"])
        frames.append(prices)

        mid = prices["mid_price"].dropna()
        hourly_change = mid.diff().abs()
        summary.append({
            "ticker": row["ticker"],
            "title": row["title"],
            "research_category": row["research_category"],
            "result": row["result"],
            "hours": len(prices),
            "pct_hours_traded": round(prices["had_trade"].mean() * 100, 1),
            "median_spread": round(prices["spread"].median(), 3),
            "first_mid": round(mid.iloc[0], 3) if len(mid) else np.nan,
            "last_mid": round(mid.iloc[-1], 3) if len(mid) else np.nan,
            "min_mid": round(mid.min(), 3) if len(mid) else np.nan,
            "max_mid": round(mid.max(), 3) if len(mid) else np.nan,
            "moves_over_5c": int((hourly_change >= 0.05).sum()),
        })

    prices_df = pd.concat(frames, ignore_index=True)
    summary_df = pd.DataFrame(summary)

    prices_df.to_csv(out_dir / "pilot_prices_hourly_2025.csv", index=False)
    summary_df.to_csv(out_dir / "pilot_price_summary.csv", index=False)

    # Nested layout from the proposal: {question, rules, prices, articles}.
    nested = {}
    for row in contracts.to_dict("records"):
        p = prices_df[prices_df["ticker"] == row["ticker"]]
        nested[row["ticker"]] = {
            "question": row["title"],
            "rules": row.get("rules_primary") or "",
            "category": row["research_category"],
            "result": row["result"],
            "prices": [
                {"t": t.isoformat(),
                 "mid": None if pd.isna(m) else float(m),
                 "volume": float(v)}
                for t, m, v in zip(p["timestamp"], p["mid_price"], p["volume"])
            ],
            "articles": [],
        }
    with open(out_dir / "pilot_dataset.json", "w") as f:
        json.dump(nested, f)

    missing = set(contracts["ticker"]) - set(summary_df["ticker"])
    print(f"\nPrice histories saved for {len(summary_df)} of {len(contracts)} "
          f"contracts in {out_dir.resolve()}")
    if missing:
        print("No data for:", ", ".join(sorted(missing)))
    return prices_df, summary_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_contracts(prices_df, contracts_df, n=4, tickers=None):
    """Plot 2025 mid prices for a few contracts, one per category by default."""
    if tickers is None:
        tickers = (
            contracts_df[contracts_df["ticker"].isin(prices_df["ticker"])]
            .groupby("research_category")["ticker"].first()
            .tolist()[:n]
        )
    titles = contracts_df.set_index("ticker")["title"]

    fig, axes = plt.subplots(len(tickers), 1,
                             figsize=(12, 3 * len(tickers)), squeeze=False)
    for ax, ticker in zip(axes[:, 0], tickers):
        p = prices_df[prices_df["ticker"] == ticker]
        ax.plot(p["timestamp"], p["mid_price"], linewidth=1)
        ax.set_title(f"{ticker}: {titles.get(ticker, '')}", fontsize=10)
        ax.set_ylabel("YES mid ($)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
