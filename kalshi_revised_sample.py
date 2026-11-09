"""
Revised pilot sample selection for the Kalshi news project.

Run from the notebook after the "Prepare contract screening variables" cell,
so that screening_df (with days_active_2025, duration_days, volume_numeric)
is in memory:

    from kalshi_revised_sample import (
        check_category_supply, select_revised_sample, compare_samples
    )

    check_category_supply(screening_df)
    revised_df = select_revised_sample(screening_df)
    compare_samples("kalshi_pilot_40_contracts.csv", revised_df)
"""

import pandas as pd

CATEGORIES = [
    "Politics and Elections",
    "Economics and Financial Markets",
    "Companies and Business",
    "Climate and Weather",
]


def _prepare(screening_df):
    df = screening_df.copy()
    df["series_ticker"] = df["event_ticker"].str.split("-").str[0]
    df["share_in_2025"] = df["days_active_2025"] / df["duration_days"]
    return df


def _apply_filters(df, min_days_2025, min_share_2025, min_volume):
    """Keep settled, liquid contracts with meaningful 2025 activity."""
    return df[
        df["result"].isin(["yes", "no"])
        & (df["days_active_2025"] >= min_days_2025)
        & (df["share_in_2025"] >= min_share_2025)
        & (df["volume_numeric"] >= min_volume)
    ]


def check_category_supply(
    screening_df,
    min_days_2025=14,
    min_share_2025=0.25,
    min_volume=1_000,
):
    """Show how many eligible contracts and series each category has."""
    df = _apply_filters(
        _prepare(screening_df), min_days_2025, min_share_2025, min_volume
    )

    supply = (
        df.groupby("research_category")
        .agg(
            eligible_contracts=("ticker", "nunique"),
            distinct_events=("event_ticker", "nunique"),
            distinct_series=("series_ticker", "nunique"),
        )
        .reindex(CATEGORIES)
        .fillna(0)
        .astype(int)
    )
    print("Eligible supply under the revised filters:")
    display(supply)

    climate = df[df["research_category"] == "Climate and Weather"]
    print("\nClimate series that survive the filters:")
    if climate.empty:
        print("  none")
    else:
        display(
            climate.groupby("series_ticker")
            .agg(
                contracts=("ticker", "nunique"),
                example_title=("title", "first"),
                median_days=("duration_days", "median"),
            )
            .sort_values("contracts", ascending=False)
        )
    return supply


def select_revised_sample(
    screening_df,
    per_category=10,
    min_days_2025=14,
    min_share_2025=0.25,
    min_volume=1_000,
    max_per_series=2,
    fill_shortfall=False,
    output_path="kalshi_pilot_revised_contracts.csv",
):
    """
    Select up to per_category contracts per research category.

    Filters: settled yes/no, at least min_days_2025 days active in 2025,
    at least min_share_2025 of the contract's life inside 2025, and
    minimum volume. Within a category, one contract per event (highest
    volume) and at most max_per_series contracts per series.

    If a category runs short, it is reported. Set fill_shortfall=True to
    top up the sample from the other categories instead.
    """
    df = _apply_filters(
        _prepare(screening_df), min_days_2025, min_share_2025, min_volume
    )

    df = (
        df.sort_values("volume_numeric", ascending=False)
        .drop_duplicates(subset=["event_ticker"])
    )
    df = df.groupby("series_ticker").head(max_per_series)

    picks = df.groupby("research_category").head(per_category)
    picks = picks[picks["research_category"].isin(CATEGORIES)]

    target_total = per_category * len(CATEGORIES)
    shortfall = target_total - len(picks)

    for category in CATEGORIES:
        n = (picks["research_category"] == category).sum()
        if n < per_category:
            print(f"Note: {category} has only {n} eligible contracts.")

    if fill_shortfall and shortfall > 0:
        leftover = df.drop(picks.index, errors="ignore")
        leftover = leftover[leftover["research_category"].isin(CATEGORIES)]
        picks = pd.concat([picks, leftover.head(shortfall)])

    picks = picks.sort_values(
        ["research_category", "volume_numeric"], ascending=[True, False]
    ).reset_index(drop=True)

    print("\nRevised sample size:", len(picks))
    display(
        picks.groupby("research_category")
        .agg(
            contracts=("ticker", "nunique"),
            series=("series_ticker", "nunique"),
            median_volume=("volume_numeric", "median"),
            median_days_2025=("days_active_2025", "median"),
        )
    )
    display(
        picks[[
            "ticker", "title", "research_category", "series_ticker",
            "days_active_2025", "share_in_2025", "volume_numeric", "result",
        ]]
    )

    picks.to_csv(output_path, index=False)
    print("Saved to", output_path)
    return picks


def compare_samples(original_csv, revised_df):
    """List which contracts were dropped from and added to the pilot."""
    original = pd.read_csv(original_csv)
    old, new = set(original["ticker"]), set(revised_df["ticker"])

    dropped = original[original["ticker"].isin(old - new)]
    added = revised_df[revised_df["ticker"].isin(new - old)]

    print(f"Kept: {len(old & new)}  Dropped: {len(dropped)}  Added: {len(added)}")
    print("\nDropped:")
    display(dropped[["ticker", "title", "research_category"]])
    print("\nAdded:")
    display(added[["ticker", "title", "research_category"]])
    return dropped, added
