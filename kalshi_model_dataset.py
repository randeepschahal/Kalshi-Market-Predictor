"""
Modeling dataset for the Kalshi news project.

Joins the hourly price series with the scored articles into one row per
(contract, checkpoint, lookback window w, horizon h):

    features  - news in the w hours up to and including the checkpoint
                (counts, VADER, FinBERT, GDELT tone, net stance if labeled)
              - market state at the checkpoint (price, recent change,
                volume, spread)
    target    - price change over the next h hours and its direction
                (up / flat / down)

Leakage rule: an article counts toward a checkpoint only if it was published
at or before that checkpoint (publication times are rounded UP to the hour),
and every market feature uses prices at or before the checkpoint.

Usage from the notebook:

    from kalshi_model_dataset import build_dataset, association_table

    dataset = build_dataset(prices_df, scored_df, model_df)
    display(association_table(dataset))
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("data/pilot")

LOOKBACKS = [1, 3, 6, 12, 24, 72]       # hours of news before a checkpoint
HORIZONS = [1, 3, 6, 12, 24]            # hours ahead for the price change
CHECKPOINT_EVERY = 6                    # hours between checkpoints
FLAT_BAND = 0.02                        # |change| below this is "flat" ($)

SENTIMENT_COLS = ["vader_compound", "finbert_net", "finbert_positive",
                  "finbert_negative", "gdelt_tone"]


# ---------------------------------------------------------------------------
# Hourly news series per contract
# ---------------------------------------------------------------------------

def _stance_value(s):
    """Map stance labels to +1 (YES), -1 (NO), 0 (neutral)."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().upper()
    return {"YES": 1.0, "NO": -1.0, "NEUTRAL": 0.0,
            "1": 1.0, "-1": -1.0, "0": 0.0}.get(s, np.nan)


def hourly_news(scored_df, relevant_only=True):
    """
    Per contract and hour: article counts and story-level sentiment sums.

    Each story (cluster_id) contributes its sentiment once, at the hour it
    first appeared for that contract, so syndicated copies do not
    multiply its weight. n_articles still counts every copy, as a measure
    of coverage intensity.
    """
    df = scored_df.copy()
    if relevant_only and "is_relevant" in df.columns:
        df = df[df["is_relevant"].astype(bool)]
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df = df.dropna(subset=["published"])
    df["hour"] = df["published"].dt.ceil("h")

    articles = (df.groupby(["ticker", "hour"]).size()
                .rename("n_articles").to_frame())

    stories = (df.sort_values("published")
               .drop_duplicates(["ticker", "cluster_id"]))
    sent_cols = [c for c in SENTIMENT_COLS if c in stories.columns]
    agg = {c: "sum" for c in sent_cols}
    stories = stories.assign(n_stories=1)
    agg["n_stories"] = "sum"
    if "stance" in stories.columns:
        stories["stance_value"] = stories["stance"].map(_stance_value)
        stories["n_stance_labeled"] = stories["stance_value"].notna().astype(int)
        stories["stance_value"] = stories["stance_value"].fillna(0)
        agg.update({"stance_value": "sum", "n_stance_labeled": "sum"})

    story_hours = stories.groupby(["ticker", "hour"]).agg(agg)
    return articles.join(story_hours, how="outer").fillna(0).reset_index()


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _contract_rows(prices, news, lookbacks, horizons, every):
    """All (checkpoint, w, h) rows for one contract on its hourly grid."""
    prices = prices.set_index("timestamp").sort_index()
    grid = pd.date_range(prices.index.min(), prices.index.max(), freq="h")
    prices = prices.reindex(grid)
    prices["mid_price"] = prices["mid_price"].ffill()
    prices["volume"] = prices["volume"].fillna(0)
    if "spread" in prices.columns:
        prices["spread"] = prices["spread"].ffill()

    news = news.set_index("hour").reindex(grid).fillna(0)
    value_cols = [c for c in news.columns if c != "ticker"]
    # Cumulative sums: window (t-w, t] = cum[t] - cum[t-w]
    cum_news = news[value_cols].cumsum().to_numpy()
    cum_vol = prices["volume"].cumsum().to_numpy()
    p = prices["mid_price"].to_numpy()
    spread = (prices["spread"].to_numpy() if "spread" in prices.columns
              else np.full(len(p), np.nan))

    n = len(grid)
    max_w, max_h = max(lookbacks), max(horizons)
    # Checkpoints need a full lookback of history; horizons are checked per h
    idx = np.arange(max_w, n, every)

    rows = []
    for w in lookbacks:
        for h in horizons:
            valid = idx[idx + h < n]
            if len(valid) == 0:
                continue
            news_w = cum_news[valid] - cum_news[valid - w]
            block = pd.DataFrame(news_w, columns=value_cols)
            block.insert(0, "checkpoint", grid[valid])
            block["lookback_h"] = w
            block["horizon_h"] = h
            block["price"] = p[valid]
            block["price_change_lookback"] = p[valid] - p[valid - w]
            block["price_change_1h"] = p[valid] - p[valid - 1]
            block["volume_lookback"] = cum_vol[valid] - cum_vol[valid - w]
            block["spread"] = spread[valid]
            block["future_price"] = p[valid + h]
            rows.append(block)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_dataset(prices_df, scored_df, contracts_df, lookbacks=LOOKBACKS,
                  horizons=HORIZONS, every=CHECKPOINT_EVERY,
                  flat_band=FLAT_BAND, relevant_only=True,
                  output_path=OUT_DIR / "model_dataset.csv"):
    """Build and save the full modeling table for every (w, h) pair."""
    prices_df = prices_df.copy()
    prices_df["timestamp"] = pd.to_datetime(prices_df["timestamp"], utc=True)
    news = hourly_news(scored_df, relevant_only=relevant_only)

    # Contracts that share the same news query move on the same articles,
    # so they share a split group (e.g. the four Fed contracts).
    if "query" in scored_df.columns:
        query_of = scored_df.groupby("ticker")["query"].first()
        groups = {q: f"G{i:02d}" for i, q in enumerate(sorted(query_of.unique()))}
        split_group = query_of.map(groups)
    else:
        split_group = pd.Series(dtype=str)

    meta = contracts_df.set_index("ticker")
    frames = []
    for ticker in contracts_df["ticker"]:
        prices = prices_df[prices_df["ticker"] == ticker]
        if prices.empty:
            print(f"  {ticker}: no prices, skipped")
            continue
        rows = _contract_rows(prices, news[news["ticker"] == ticker]
                              .drop(columns="ticker"),
                              lookbacks, horizons, every)
        if rows.empty:
            continue
        rows.insert(0, "ticker", ticker)
        rows.insert(1, "research_category", meta.at[ticker, "research_category"])
        rows.insert(2, "event_ticker", meta.at[ticker, "event_ticker"])
        rows.insert(3, "split_group", split_group.get(ticker, ticker))
        frames.append(rows)

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["price", "future_price"])

    # Averages from window sums (story-level, so syndication is not counted twice)
    for col in SENTIMENT_COLS:
        if col in data.columns:
            data[f"mean_{col}"] = np.where(data["n_stories"] > 0,
                                           data[col] / data["n_stories"],
                                           0.0)
            data = data.drop(columns=col)
    if "stance_value" in data.columns:
        data["net_stance"] = np.where(data["n_stance_labeled"] > 0,
                                      data["stance_value"]
                                      / data["n_stance_labeled"], 0.0)
        data = data.drop(columns="stance_value")
    data["has_news"] = data["n_stories"] > 0
    data["log_articles"] = np.log1p(data["n_articles"])

    # Targets
    data["price_change"] = data["future_price"] - data["price"]
    data["direction"] = np.select(
        [data["price_change"] > flat_band, data["price_change"] < -flat_band],
        ["up", "down"], default="flat")
    # Prices pinned near 0 or 1 have little room to move; flag rather than drop
    data["near_bound"] = (data["price"] <= 0.03) | (data["price"] >= 0.97)

    data = data.sort_values(["lookback_h", "horizon_h", "checkpoint", "ticker"])
    data = data.reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    one = data[(data["lookback_h"] == lookbacks[0])
               & (data["horizon_h"] == horizons[0])]
    print(f"Saved {len(data):,} rows to {output_path}")
    print(f"{data['ticker'].nunique()} contracts, {len(one):,} checkpoints per "
          f"(w, h) pair, {len(lookbacks)} lookbacks x {len(horizons)} horizons")
    print("Share of checkpoints with relevant news in the window:")
    display(data.groupby("lookback_h")["has_news"].mean().round(3).to_frame())
    print("Direction balance (all pairs):")
    display(data["direction"].value_counts(normalize=True).round(3).to_frame())
    return data


# ---------------------------------------------------------------------------
# Stage 1 of the proposal: association
# ---------------------------------------------------------------------------

def association_table(data, exclude_near_bound=True):
    """
    Spearman correlations between news features and the next-h price change,
    for each (lookback, horizon) pair. Directional features (sentiment, stance)
    are correlated with the signed change; news volume with its size.
    """
    df = data[~data["near_bound"]] if exclude_near_bound else data
    df = df.assign(abs_change=df["price_change"].abs())

    signed = [c for c in ["mean_vader_compound", "mean_finbert_net",
                          "mean_gdelt_tone", "net_stance"] if c in df.columns]
    rows = []
    for (w, h), g in df.groupby(["lookback_h", "horizon_h"]):
        news = g[g["has_news"]]
        row = {"lookback_h": w, "horizon_h": h, "rows": len(g),
               "rows_with_news": len(news),
               "log_articles_vs_abs_change":
                   g["log_articles"].corr(g["abs_change"], method="spearman")}
        for c in signed:
            row[f"{c}_vs_change"] = (news[c].corr(news["price_change"],
                                                  method="spearman")
                                     if len(news) > 30 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).round(3)
