"""
GDELT article collection through Google BigQuery (GKG 2.x tables).

Replaces the DOC API collector: one scan of 2025 returns every matching
article for all contracts at once, with no rate limit and no 250-per-request
cap. Output has the same columns as kalshi_gdelt_news.collect_articles, so
the scoring step (kalshi_news_scoring) works unchanged.

One-time setup (see the chat for details):
    pip install google-cloud-bigquery db-dtypes
    gcloud auth application-default login

Usage from the notebook:

    from kalshi_gdelt_bigquery import estimate_cost, collect_articles_bq

    PROJECT = "your-gcp-project-id"
    estimate_cost(news_df, PROJECT)                 # free dry run, shows GB scanned
    articles_df, news_summary_df = collect_articles_bq(news_df, PROJECT)

Matching is on article headlines (the <PAGE_TITLE> GDELT stores in the
Extras field). Each contract has keyword groups: a headline must contain at
least one term from every group, and be published inside the contract's
2025 window. The raw matches are saved to disk, so changing terms later
re-filters locally without another BigQuery scan.
"""

import hashlib
import html
import re
from pathlib import Path

import pandas as pd

STUDY_START = pd.Timestamp("2025-01-01", tz="UTC")
STUDY_END = pd.Timestamp("2026-01-01", tz="UTC")
OUT_DIR = Path("data/pilot")
RAW_PATH = OUT_DIR / "gkg_headline_matches_2025.csv"

# Safety cap: the query fails instead of running if it would scan more
# than this. BigQuery's free tier is 1 TB of scanning per month.
MAX_GB_BILLED = 900

# ---------------------------------------------------------------------------
# Headline terms. Each contract: list of groups; every group must match
# (AND), any term within a group may match (OR). Lowercase, whole words.
# ---------------------------------------------------------------------------

FED = [["federal reserve", "fed", "fomc", "powell"],
       ["rate", "rates", "cut", "cuts", "hike", "hold", "holds", "policy",
        "meeting", "inflation"]]

CONTRACT_TERMS = {
    # Companies and Business
    "KXSUPERBOWLAD-SB2026-ANTHROPIC": [["anthropic", "claude"],
                                       ["super bowl", "big game"]],
    "KXMUSKPACKAGEVOTE-26JAN01": [["musk", "tesla"],
                                  ["pay package", "compensation package",
                                   "pay deal", "pay plan", "pay vote",
                                   "compensation plan"]],
    "KXOAIPROFIT-26-JAN01": [["openai"],
                             ["nonprofit", "non-profit", "for-profit",
                              "restructure", "restructuring", "restructures",
                              "public benefit", "recapitalization"]],
    "KXLEAVECRACKERBARREL-26JAN": [["cracker barrel"]],
    "KXAAPLPRICE17OVER16-25-799.01": [["iphone 17"],
                                      ["price", "prices", "pricing", "cost",
                                       "costs", "tariff", "tariffs"]],
    "KXAAPLPRICEFOLD-27-2000": [["foldable iphone", "iphone fold"]],
    "KXELONSUEAPPLE-26": [["musk", "xai"], ["apple"],
                          ["lawsuit", "sue", "sues", "suing", "antitrust"]],

    # Economics and Financial Markets
    "KXFEDDECISION-25SEP-H0": FED,
    "KXRATECUTCOUNT-25DEC31-T3": FED,
    "FED-25MAY-T4.25": FED,
    "FED-25SEP-T4.25": FED,
    "KXINXY-25DEC31-B7100": [["s&p 500", "s&p", "wall street",
                              "stock market"]],
    "KXCPIYOY-25APR-T2.5": [["cpi", "consumer price index", "inflation"]],
    "KXAAAGASM-25JUN30-3.30": [["gas prices", "gas price", "gasoline prices",
                                "pump prices", "prices at the pump"]],

    # Politics and Elections
    "CONTROLH-2026-R": [["house majority", "control of the house",
                         "house control", "midterm", "midterms",
                         "house republicans", "house democrats"]],
    "CONTROLS-2026-R": [["senate majority", "control of the senate",
                         "senate control", "senate map", "senate races",
                         "senate seats", "senate republicans",
                         "senate democrats"]],
    "SENATETX-26-R": [["paxton", "cornyn", "talarico", "allred",
                       "texas senate"],
                      ["senate", "primary", "race", "poll", "campaign",
                       "runoff"]],
    "SENATEMI-26-R": [["michigan"], ["senate"]],
    "KXFEDCHAIRNOM-29-JS": [["fed chair", "federal reserve chair",
                             "fed chairman", "powell successor",
                             "judy shelton", "next fed chair"]],
    "KXMAYORLA-26-KBAS": [["karen bass", "los angeles mayor", "l.a. mayor",
                           "la mayor"]],
    "KXGOVCA-26-SHIL": [["california governor", "governor of california",
                         "california gubernatorial", "steve hilton"]],
    "KXGOVFLNOMR-26-BD": [["byron donalds", "florida governor",
                           "florida gubernatorial"]],
}

# Contracts about US-specific outcomes: drop articles from outlets GDELT
# knows to be outside the US (outlets with unknown country are kept).
US_ONLY = {
    "KXFEDDECISION-25SEP-H0", "KXRATECUTCOUNT-25DEC31-T3", "FED-25MAY-T4.25",
    "FED-25SEP-T4.25", "KXCPIYOY-25APR-T2.5", "KXAAAGASM-25JUN30-3.30",
    "CONTROLH-2026-R", "CONTROLS-2026-R", "SENATETX-26-R", "SENATEMI-26-R",
    "KXFEDCHAIRNOM-29-JS", "KXMAYORLA-26-KBAS", "KXGOVCA-26-SHIL",
    "KXGOVFLNOMR-26-BD",
}


# ---------------------------------------------------------------------------
# Regex helpers (shared by the SQL prefilter and the local matching)
# ---------------------------------------------------------------------------

def _term_regex(term):
    return r"\b" + term.replace(".", r"\.") + r"\b"


def _group_regex(group):
    return "(?:" + "|".join(_term_regex(t) for t in group) + ")"


def _prefilter_regex(terms):
    """Any headline matching the first group of any contract."""
    firsts = sorted({t for groups in terms.values() for t in groups[0]})
    return _group_regex(firsts)


# ---------------------------------------------------------------------------
# BigQuery
# ---------------------------------------------------------------------------

def build_query(terms=None, start=STUDY_START, end=STUDY_END):
    terms = terms or CONTRACT_TERMS
    # Partitions are by ingestion day; one extra day catches late arrivals.
    p_start = start.strftime("%Y-%m-%d")
    p_end = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return f"""
    WITH gkg AS (
      SELECT
        DATE,
        SourceCommonName,
        DocumentIdentifier,
        V2Tone,
        REPLACE(REPLACE(
          REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>'),
          '&amp;', '&'), '&#39;', "'") AS title
      FROM `gdelt-bq.gdeltv2.gkg_partitioned`
      WHERE _PARTITIONTIME >= TIMESTAMP("{p_start}")
        AND _PARTITIONTIME <  TIMESTAMP("{p_end}")
        AND (TranslationInfo IS NULL OR TranslationInfo = '')
    ),
    countries AS (
      SELECT Domain, ANY_VALUE(FIPS) AS FIPS
      FROM `gdelt-bq.extra.sourcesbycountry`
      GROUP BY Domain
    )
    SELECT g.DATE, g.SourceCommonName, g.DocumentIdentifier, g.V2Tone,
           g.title, c.FIPS AS source_country
    FROM gkg g
    LEFT JOIN countries c ON g.SourceCommonName = c.Domain
    WHERE g.title IS NOT NULL
      AND REGEXP_CONTAINS(LOWER(g.title), r'{_prefilter_regex(terms)}')
    """


def _client(project):
    from google.cloud import bigquery
    return bigquery.Client(project=project)


def estimate_cost(contracts_df=None, project=None, terms=None, client=None):
    """Dry run: report how much data the query would scan. Costs nothing."""
    from google.cloud import bigquery
    client = client or _client(project)
    job = client.query(
        build_query(terms),
        job_config=bigquery.QueryJobConfig(dry_run=True,
                                           use_query_cache=False),
    )
    gb = job.total_bytes_processed / 1e9
    print(f"Query would scan {gb:,.0f} GB "
          f"({gb / 1000:.0%} of the 1 TB monthly free tier).")
    return gb


def fetch_raw_matches(project, terms=None, refresh=False, client=None):
    """Run the scan once and save the matches; later calls reuse the file."""
    if RAW_PATH.exists() and not refresh:
        print(f"Using saved matches in {RAW_PATH}")
        return pd.read_csv(RAW_PATH)

    from google.cloud import bigquery
    client = client or _client(project)
    config = bigquery.QueryJobConfig(
        maximum_bytes_billed=int(MAX_GB_BILLED * 1e9))
    print("Running BigQuery scan (usually 1 to 5 minutes)...")
    raw = client.query(build_query(terms), job_config=config).to_dataframe()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(raw):,} headline matches to {RAW_PATH}")
    return raw


# ---------------------------------------------------------------------------
# Local matching to contracts
# ---------------------------------------------------------------------------

def _normalise_title(title):
    title = str(title).lower()
    title = re.sub(r"\s+[-|–]\s+[^-|–]+$", "", title)
    title = re.sub(r"[^a-z0-9 ]", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def _cluster_id(title):
    return hashlib.sha1(_normalise_title(title).encode()).hexdigest()[:12]


def match_contracts(raw, contracts_df, terms=None):
    """Assign headline matches to contracts by terms, window and country."""
    terms = terms or CONTRACT_TERMS
    raw = raw.copy()
    raw["title"] = raw["title"].astype(str).map(html.unescape).str.strip()
    raw["published"] = pd.to_datetime(raw["DATE"].astype("int64").astype(str),
                                      format="%Y%m%d%H%M%S", utc=True,
                                      errors="coerce")
    raw["gdelt_tone"] = pd.to_numeric(
        raw["V2Tone"].astype(str).str.split(",").str[0], errors="coerce")
    lower = raw["title"].str.lower()

    contracts = contracts_df.copy()
    for col in ["created_time", "effective_end_time"]:
        contracts[col] = pd.to_datetime(contracts[col], utc=True)

    frames = []
    for c in contracts.to_dict("records"):
        groups = terms.get(c["ticker"])
        if groups is None:
            continue
        start = max(c["created_time"], STUDY_START)
        end = min(c["effective_end_time"], STUDY_END)

        mask = raw["published"].between(start, end)
        for group in groups:
            mask &= lower.str.contains(_group_regex(group), regex=True)
        if c["ticker"] in US_ONLY:
            mask &= raw["source_country"].isna() | (raw["source_country"] == "US")

        hits = raw.loc[mask]
        frames.append(pd.DataFrame({
            "ticker": c["ticker"],
            "research_category": c["research_category"],
            "query": " AND ".join("(" + " OR ".join(g) + ")" for g in groups),
            "url": hits["DocumentIdentifier"],
            "title": hits["title"],
            "domain": hits["SourceCommonName"],
            "language": "English",
            "source_country": hits["source_country"],
            "published": hits["published"],
            "gdelt_tone": hits["gdelt_tone"],
        }))

    articles = pd.concat(frames, ignore_index=True)
    articles = articles.drop_duplicates(["ticker", "url"])
    articles["cluster_id"] = articles["title"].map(_cluster_id)
    return articles.sort_values(["ticker", "published"]).reset_index(drop=True)


def collect_articles_bq(contracts_df, project=None, terms=None, refresh=False,
                        client=None):
    """Scan (or reuse saved scan), match to contracts, save like the DOC version."""
    terms = terms or CONTRACT_TERMS
    missing = sorted(set(contracts_df["ticker"]) - set(terms))
    if missing:
        print("No terms defined, skipping:", ", ".join(missing))

    raw = fetch_raw_matches(project, terms, refresh=refresh, client=client)
    articles = match_contracts(raw, contracts_df, terms)

    summary = (articles.groupby("ticker")
               .agg(articles=("url", "count"),
                    unique_stories=("cluster_id", "nunique"),
                    first_article=("published", "min"),
                    last_article=("published", "max"))
               .reindex([t for t in contracts_df["ticker"] if t in terms])
               .reset_index())
    summary[["articles", "unique_stories"]] = (
        summary[["articles", "unique_stories"]].fillna(0).astype(int))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    articles.to_csv(OUT_DIR / "pilot_articles.csv", index=False)
    summary.to_csv(OUT_DIR / "pilot_news_summary.csv", index=False)
    print(f"Saved {len(articles):,} articles for {summary['articles'].gt(0).sum()} "
          f"contracts to {OUT_DIR.resolve()}")
    return articles, summary
