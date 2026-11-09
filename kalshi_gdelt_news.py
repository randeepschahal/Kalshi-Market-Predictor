"""
GDELT article collection for the Kalshi pilot contracts.

Run from the notebook after the price-history cell (needs revised_df):

    from kalshi_gdelt_news import CONTRACT_QUERIES, collect_articles, clear_empty_cache

    # Quick test on two contracts first
    test_articles, test_summary = collect_articles(
        revised_df, tickers=["KXOAIPROFIT-26-JAN01", "KXAAAGASM-25JUN30-3.30"]
    )

    # Then the full set (resumes from cache if interrupted)
    articles_df, news_summary_df = collect_articles(revised_df)

GDELT's DOC API returns at most 250 articles per request, and its article
list only covers the last 3 months of any requested range, so each
contract's 2025 window is queried in weekly slices. GDELT asks for no more
than one request every 5 seconds; responses are cached to disk so reruns
skip windows already fetched, and contracts sharing a query share requests.
"""

import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
STUDY_START = pd.Timestamp("2025-01-01", tz="UTC")
STUDY_END = pd.Timestamp("2026-01-01", tz="UTC")
REQUEST_GAP_SECONDS = 5.5
MAX_RECORDS = 250

# ---------------------------------------------------------------------------
# Queries: one per contract. Edit freely. GDELT syntax: quoted phrases,
# OR-lists inside parentheses, space means AND. Very short or very common
# words are rejected by GDELT, so prefer full names ("Federal Reserve").
# ---------------------------------------------------------------------------

FED_RATES = '"Federal Reserve" ("rate cut" OR "interest rates" OR FOMC OR Powell)'

CONTRACT_QUERIES = {
    # Companies and Business
    "KXSUPERBOWLAD-SB2026-ANTHROPIC":
        'Anthropic ("Super Bowl" OR "Big Game")',
    "KXMUSKPACKAGEVOTE-26JAN01":
        '(Musk OR Tesla) ("pay package" OR "compensation package")',
    "KXOAIPROFIT-26-JAN01":
        'OpenAI (nonprofit OR "for-profit" OR restructuring '
        'OR "public benefit corporation")',
    "KXLEAVECRACKERBARREL-26JAN":
        '"Cracker Barrel" (Masino OR CEO OR logo)',
    "KXAAPLPRICE17OVER16-25-799.01":
        '"iPhone 17" (price OR pricing OR cost)',
    "KXAAPLPRICEFOLD-27-2000":
        '("foldable iPhone" OR "iPhone Fold")',
    "KXELONSUEAPPLE-26":
        '(Musk OR xAI) Apple (lawsuit OR sue OR antitrust)',

    # Economics and Financial Markets
    "KXFEDDECISION-25SEP-H0": FED_RATES,
    "KXRATECUTCOUNT-25DEC31-T3": FED_RATES,
    "FED-25MAY-T4.25": FED_RATES,
    "FED-25SEP-T4.25": FED_RATES,
    "KXINXY-25DEC31-B7100":
        '"S&P 500" (record OR rally OR selloff OR outlook)',
    "KXCPIYOY-25APR-T2.5":
        '("consumer price index" OR "CPI inflation" OR "inflation rate")',
    "KXAAAGASM-25JUN30-3.30":
        '("gas prices" OR "gasoline prices")',

    # Politics and Elections
    "CONTROLH-2026-R":
        '("House majority" OR "control of the House" OR "midterm elections")',
    "CONTROLS-2026-R":
        '("Senate majority" OR "control of the Senate" OR "Senate map")',
    "SENATETX-26-R":
        '("Texas Senate" OR Paxton OR Cornyn OR Talarico)',
    "SENATEMI-26-R":
        'Michigan ("Senate race" OR "Senate seat")',
    "KXFEDCHAIRNOM-29-JS":
        '("Fed chair" OR "Federal Reserve chair") (nominee OR nominate OR Shelton)',
    "KXMAYORLA-26-KBAS":
        '("Karen Bass" OR "Los Angeles mayor")',
    "KXGOVCA-26-SHIL":
        '("California governor" OR "Steve Hilton")',
    "KXGOVFLNOMR-26-BD":
        '("Byron Donalds" OR "Florida governor")',
}


# ---------------------------------------------------------------------------
# Requests with rate limiting and caching
# ---------------------------------------------------------------------------

_last_request = [0.0]


def _wait_for_slot():
    gap = time.time() - _last_request[0]
    if gap < REQUEST_GAP_SECONDS:
        time.sleep(REQUEST_GAP_SECONDS - gap)
    _last_request[0] = time.time()


def _fmt(ts):
    return ts.strftime("%Y%m%d%H%M%S")


def _cache_path(cache_dir, query, start, end):
    key = hashlib.sha1(f"{query}|{_fmt(start)}|{_fmt(end)}".encode()).hexdigest()
    return cache_dir / f"{key}.json"


def fetch_window(query, start, end, cache_dir, max_retries=4):
    """Articles for one query and time window; cached on disk."""
    path = _cache_path(cache_dir, query, start, end)
    if path.exists():
        return json.loads(path.read_text())

    params = {
        "query": f"{query} sourcelang:english",
        "mode": "ArtList",
        "format": "json",
        "maxrecords": MAX_RECORDS,
        "sort": "HybridRel",
        "startdatetime": _fmt(start),
        "enddatetime": _fmt(end),
    }

    articles, error, parsed = [], "no response after retries", False
    for attempt in range(max_retries):
        _wait_for_slot()
        try:
            response = requests.get(GDELT_URL, params=params, timeout=60)
        except requests.RequestException as exc:
            error = str(exc)
            time.sleep(10 * (attempt + 1))
            continue

        text = response.text.strip()
        if response.status_code == 429 or "limit requests" in text.lower():
            error = "rate limited on every attempt"
            time.sleep(15 * (attempt + 1))
            continue
        if not text:
            # Usually means no matches, but GDELT also sends empty bodies
            # when overloaded, so this is returned without caching.
            return []
        try:
            articles = json.loads(text, strict=False).get("articles", [])
            parsed = True
            break
        except json.JSONDecodeError:
            # Plain-text messages, e.g. a rejected query
            error = text[:200]
            break

    if not parsed:
        print(f"    GDELT message ({_fmt(start)[:8]}): {error}")
        return []  # not cached, so it is retried next run

    path.write_text(json.dumps(articles))
    return articles


def clear_empty_cache(cache_dir="data/pilot/gdelt_cache"):
    """Delete cached windows that hold no articles so they are fetched again."""
    removed = 0
    for path in Path(cache_dir).glob("*.json"):
        if path.read_text().strip() in ("", "[]"):
            path.unlink()
            removed += 1
    print(f"Removed {removed} empty cached windows from {cache_dir}")
    return removed


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def _normalise_title(title):
    title = str(title).lower()
    title = re.sub(r"\s+[-|–]\s+[^-|–]+$", "", title)   # drop " - Source"
    title = re.sub(r"[^a-z0-9 ]", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def _cluster_id(title):
    return hashlib.sha1(_normalise_title(title).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def _weekly_windows(start, end, days=7):
    edges = list(pd.date_range(start, end, freq=f"{days}D"))
    if edges[-1] < end:
        edges.append(end)
    return list(zip(edges[:-1], edges[1:]))


def collect_articles(
    contracts_df,
    queries=None,
    tickers=None,
    window_days=7,
    out_dir="data/pilot",
):
    """Collect, deduplicate and save GDELT articles for each contract."""
    queries = queries or CONTRACT_QUERIES
    out_dir = Path(out_dir)
    cache_dir = out_dir / "gdelt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    contracts = contracts_df.copy()
    for col in ["created_time", "effective_end_time"]:
        contracts[col] = pd.to_datetime(contracts[col], utc=True)
    if tickers is not None:
        contracts = contracts[contracts["ticker"].isin(tickers)]

    missing = sorted(set(contracts["ticker"]) - set(queries))
    if missing:
        print("No query defined, skipping:", ", ".join(missing))
    contracts = contracts[contracts["ticker"].isin(queries)]

    rows, summary = [], []
    for i, c in enumerate(contracts.to_dict("records"), start=1):
        query = queries[c["ticker"]]
        start = max(c["created_time"], STUDY_START).floor("D")
        end = min(c["effective_end_time"], STUDY_END)
        windows = _weekly_windows(start, end, window_days)
        print(f"{i}/{len(contracts)}: {c['ticker']} "
              f"({len(windows)} windows)")

        capped = 0
        for w_start, w_end in windows:
            articles = fetch_window(query, w_start, w_end, cache_dir)
            capped += len(articles) >= MAX_RECORDS
            for a in articles:
                rows.append({
                    "ticker": c["ticker"],
                    "research_category": c["research_category"],
                    "query": query,
                    "url": a.get("url"),
                    "title": a.get("title"),
                    "domain": a.get("domain"),
                    "language": a.get("language"),
                    "source_country": a.get("sourcecountry"),
                    "seendate": a.get("seendate"),
                })

        summary.append({"ticker": c["ticker"], "windows": len(windows),
                        "windows_capped_at_250": int(capped)})

    articles_df = pd.DataFrame(rows)
    if articles_df.empty:
        print("No articles retrieved.")
        return articles_df, pd.DataFrame(summary)

    articles_df["published"] = pd.to_datetime(
        articles_df["seendate"], format="%Y%m%dT%H%M%SZ",
        utc=True, errors="coerce")
    articles_df = articles_df.drop(columns="seendate")

    # Same URL counted once per contract; syndicated copies share a cluster
    articles_df = articles_df.drop_duplicates(["ticker", "url"])
    articles_df["cluster_id"] = articles_df["title"].apply(_cluster_id)
    articles_df = articles_df.sort_values(["ticker", "published"]) \
                             .reset_index(drop=True)

    counts = (
        articles_df.groupby("ticker")
        .agg(articles=("url", "count"),
             unique_stories=("cluster_id", "nunique"),
             first_article=("published", "min"),
             last_article=("published", "max"))
        .reset_index()
    )
    summary_df = pd.DataFrame(summary).merge(counts, on="ticker", how="left")
    summary_df[["articles", "unique_stories"]] = (
        summary_df[["articles", "unique_stories"]].fillna(0).astype(int))

    articles_df.to_csv(out_dir / "pilot_articles.csv", index=False)
    summary_df.to_csv(out_dir / "pilot_news_summary.csv", index=False)
    print(f"\nSaved {len(articles_df)} articles to {out_dir.resolve()}")
    return articles_df, summary_df
